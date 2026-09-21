"""Bounded-age asynchronous NRC deployment with an explicit action contract.

Double buffering retains row order but does not provide server-side RTC prefix
conditioning. Reject late results and bound each row against its observation-
relative scheduled time. The interface interprets absolute or residual targets.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any

from camera_preview import CameraPreviewPolicy
from interface import InterfaceConfig
from interface import NrcArmO6Interface
from interface import validate_task_prompt
from interface import warmup_policy
import numpy as np
from openpi_client import websocket_client_policy

ACTION_CONTRACT = NrcArmO6Interface.ACTION_CONTRACT
ACTION_DIM = NrcArmO6Interface.ACTION_DIM
# The checkpoint was trained on 20 Hz demonstrations, so one 20-row chunk is one
# second of demonstrated motion. Execution now runs at that same 20 Hz. Running
# the chunk at 30 Hz consumed it in 0.667 s: the (un-rate-limited) gripper
# timeline advanced 1.5x faster than the demonstration while every joint row is
# capped at joint_target_max_speed_rad_s * dt, so the arm fell progressively
# further behind the gripper's schedule.
POLICY_CONTROL_HZ = 20.0
CONTROL_HZ = 20.0
PREVIEW_ONLY_PERIOD_S = 0.5


@dataclass(frozen=True)
class RTCDeploymentConfig:
    control_hz: float = CONTROL_HZ
    request_lead_steps: int = 6
    inference_timeout_s: float = 30.0
    max_observation_age_s: float = 0.35

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not np.isfinite(self.max_observation_age_s) or self.max_observation_age_s <= 0:
            raise ValueError("max_observation_age_s must be positive")
        if self.request_lead_steps < 1:
            raise ValueError("request_lead_steps must be >= 1")
        if not np.isfinite(self.inference_timeout_s) or self.inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be positive")


@dataclass
class _InferenceResult:
    request_id: int
    actions: np.ndarray
    latency_s: float
    observation_timestamp_ns: int
    request_started_ns: int
    observation_latency_s: float
    ready_ns: int


class AsyncDeltaRTCController:
    """Asynchronous double-buffer controller; the interface owns action semantics."""

    def __init__(
        self,
        robot: NrcArmO6Interface,
        policy: websocket_client_policy.WebsocketClientPolicy | CameraPreviewPolicy,
        prompt: str,
        config: RTCDeploymentConfig,
    ):
        if not str(prompt).strip():
            raise ValueError("prompt must not be empty")

        self.robot = robot
        self.policy = policy
        self.prompt = str(prompt)
        self.config = config

        self._stop = threading.Event()
        self._request_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._lock = threading.RLock()

        self._results: queue.Queue[_InferenceResult] = queue.Queue()
        self._active_actions: deque[tuple[np.ndarray, int]] = deque()
        self._pending_chunk: _InferenceResult | None = None
        self._request_started_ns = 0

        self._request_in_flight = False
        self._next_request_id = 1
        self._worker_error: BaseException | None = None

        self._control_steps = 0
        self._requests = 0
        self._accepted_chunks = 0
        self._hold_steps = 0
        self._buffer_underrun_steps = 0
        self._control_overruns = 0
        self._chunk_handoffs = 0
        self._inference_latencies_s: list[float] = []
        self._request_latencies_s: list[float] = []
        self._chunk_shape: tuple[int, int] | None = None

    def _validate_chunk(self, result: Any, request_id: int) -> np.ndarray:
        if not isinstance(result, dict):
            raise RuntimeError(
                f"request {request_id}: policy result must be dict, "
                f"got {type(result).__name__}"
            )
        if "actions" not in result:
            raise RuntimeError(
                f"request {request_id}: result missing 'actions'; "
                f"keys={list(result.keys())}"
            )

        chunk = np.asarray(result["actions"], dtype=np.float32)
        if (
            chunk.ndim != 2
            or chunk.shape[0] == 0
            or chunk.shape[1] != ACTION_DIM
            or not np.all(np.isfinite(chunk))
        ):
            raise RuntimeError(
                f"request {request_id}: invalid action chunk {chunk.shape}; "
                f"expected [T,{ACTION_DIM}] finite values"
            )
        return np.ascontiguousarray(chunk)

    def _schedule_request(self) -> bool:
        with self._lock:
            if (
                self._stop.is_set()
                or self._request_in_flight
                or self._pending_chunk is not None
            ):
                return False
            self._request_in_flight = True
            self._request_started_ns = time.monotonic_ns()
            self._requests += 1
        logging.info(
            "RTC request scheduled: request=%d active_remaining=%d nominal_lead_ms=%.1f",
            self._requests, len(self._active_actions),
            len(self._active_actions) * 1000.0 / self.config.control_hz,
        )
        self._request_event.set()
        return True

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            self._request_event.wait(0.1)
            if self._stop.is_set():
                return
            if not self._request_event.is_set():
                continue
            self._request_event.clear()

            with self._lock:
                if not self._request_in_flight:
                    continue
                request_id = self._next_request_id
                self._next_request_id += 1
                request_started_ns = self._request_started_ns

            try:
                observation_started = time.monotonic()
                observation = self.robot.get_observation(self.prompt)
                observation_latency_s = time.monotonic() - observation_started
                observation_timestamp_ns = int(observation["timestamp_ns"])

                started = time.monotonic()
                result = self.policy.infer(observation)
                latency_s = time.monotonic() - started

                chunk = self._validate_chunk(result, request_id)

                self._results.put(
                    _InferenceResult(
                        request_id=request_id,
                        actions=chunk,
                        latency_s=latency_s,
                        observation_timestamp_ns=observation_timestamp_ns,
                        request_started_ns=request_started_ns,
                        observation_latency_s=observation_latency_s,
                        ready_ns=time.monotonic_ns(),
                    )
                )
            except BaseException as exc:
                with self._lock:
                    self._worker_error = exc
                self._stop.set()
                return

    def _raise_worker_error(self) -> None:
        with self._lock:
            exc = self._worker_error
        if exc is not None:
            raise RuntimeError(
                f"RTC inference worker failed: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _chunk_stats(chunk: np.ndarray) -> str:
        # Hand dims 6..11 are raw O6 register targets (0..255); the demonstrated
        # open pose is ~[157, 95, 175] and the grasp pose ~[78, 85, 123].
        # Logging them makes "the model never asked for a grasp" visible in the
        # terminal instead of only on the hardware.
        return (
            f"first_joint_target_deg={np.round(np.rad2deg(chunk[0, :6]), 3).tolist()} "
            f"first_hand={np.round(chunk[0, 6:12], 1).tolist()} "
            f"thumb_min={float(np.min(chunk[:, 6])):.1f} "
            f"close_steps={int(np.sum(chunk[:, 6] < 100.0))}/{chunk.shape[0]}"
        )

    def _receive_results(self) -> None:
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                return

            if self._pending_chunk is not None:
                raise RuntimeError(
                    "internal RTC error: received a new chunk while another "
                    "pending chunk has not been consumed"
                )

            self._check_age(item.observation_timestamp_ns)
            shape = tuple(int(v) for v in item.actions.shape)
            if self._chunk_shape is None:
                self._chunk_shape = shape
            elif shape != self._chunk_shape:
                raise RuntimeError(
                    "policy action chunk shape changed: "
                    f"first={self._chunk_shape}, current={shape}"
                )

            self._pending_chunk = item
            with self._lock:
                self._request_in_flight = False
            self._accepted_chunks += 1
            self._inference_latencies_s.append(item.latency_s)
            received_ns = time.monotonic_ns()
            request_latency_s = (received_ns - item.request_started_ns) / 1e9
            self._request_latencies_s.append(request_latency_s)

            logging.info(
                "RTC chunk ready: request=%d shape=%s latency=%.1fms "
                "observation_ms=%.1f request_to_ready_ms=%.1f delivery_ms=%.1f "
                "end_to_end_ms=%.1f observation_age_ms=%.1f "
                "active_remaining=%d buffer_underrun_steps=%d %s",
                item.request_id,
                shape,
                item.latency_s * 1000.0,
                item.observation_latency_s * 1000.0,
                (item.ready_ns - item.request_started_ns) / 1e6,
                (received_ns - item.ready_ns) / 1e6,
                request_latency_s * 1000.0,
                (received_ns - item.observation_timestamp_ns) / 1e6,
                len(self._active_actions),
                self._buffer_underrun_steps,
                self._chunk_stats(item.actions),
            )

    def _promote_pending_if_boundary(self) -> None:
        if self._active_actions or self._pending_chunk is None:
            return

        item = self._pending_chunk
        self._check_age(item.observation_timestamp_ns)
        chunk = item.actions
        self._pending_chunk = None

        for index, action in enumerate(chunk):
            scheduled_ns = item.observation_timestamp_ns + round(index * 1e9 / self.config.control_hz)
            self._active_actions.append((action.copy(), scheduled_ns))

        self._chunk_handoffs += 1
        logging.info(
            "RTC chunk activated: actions=%d queue=%d nominal_chunk_ms=%.1f",
            len(chunk),
            len(self._active_actions),
            len(chunk) * 1000.0 / self.config.control_hz,
        )

    def _check_age(self, timestamp_ns: int) -> None:
        age_s = (time.monotonic_ns() - timestamp_ns) / 1e9
        if not 0 <= age_s <= self.config.max_observation_age_s:
            raise TimeoutError(f"RTC action/observation age invalid: {age_s:.3f}s")

    def _should_request(self) -> bool:
        with self._lock:
            in_flight = self._request_in_flight

        return (
            not self._stop.is_set()
            and not in_flight
            and self._pending_chunk is None
            and len(self._active_actions) <= self.config.request_lead_steps
        )

    def run(self, duration_s: float | None = None) -> dict[str, Any]:
        if duration_s is not None:
            duration_s = float(duration_s)
            if not np.isfinite(duration_s) or duration_s <= 0:
                raise ValueError("duration_s must be positive")

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="OpenPI-Delta-RTC-Inference",
            daemon=True,
        )
        self._worker.start()

        # Until a fresh result arrives, resend one fixed joint/hand target.
        self._schedule_request()
        logging.info(
            "RTC waiting for a fresh initial policy chunk"
        )

        period_s = 1.0 / self.config.control_hz
        started_at = time.monotonic()
        next_tick = started_at

        try:
            while not self._stop.is_set():
                self._raise_worker_error()
                self._receive_results()
                self._promote_pending_if_boundary()
                if self._request_in_flight:
                    self._check_age(self._request_started_ns)

                if duration_s is not None and time.monotonic() - started_at >= duration_s:
                    break

                if self._should_request():
                    self._schedule_request()

                if self._active_actions:
                    action, scheduled_ns = self._active_actions.popleft()
                    self._check_age(scheduled_ns)
                    self.robot.apply_action(action, observation_timestamp_ns=scheduled_ns)
                else:
                    self.robot.track_or_hold()
                    self._hold_steps += 1
                    if self._chunk_handoffs:
                        self._buffer_underrun_steps += 1
                        logging.warning(
                            "RTC buffer underrun: steps=%d request_elapsed_ms=%.1f",
                            self._buffer_underrun_steps,
                            (time.monotonic_ns() - self._request_started_ns) / 1e6,
                        )

                self._control_steps += 1

                next_tick += period_s
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    self._control_overruns += 1
                    # Never catch up by bursting multiple ServoJ commands.
                    next_tick = time.monotonic()

            self._raise_worker_error()
            return self.diagnostics()
        finally:
            self._stop.set()
            self._request_event.set()

    def diagnostics(self) -> dict[str, Any]:
        values = np.asarray(self._inference_latencies_s, dtype=np.float64)
        if values.size:
            mean_ms = float(np.mean(values) * 1000.0)
            p50_ms = float(np.percentile(values, 50) * 1000.0)
            p95_ms = float(np.percentile(values, 95) * 1000.0)
            max_ms = float(np.max(values) * 1000.0)
        else:
            mean_ms = p50_ms = p95_ms = max_ms = float("nan")

        with self._lock:
            in_flight = self._request_in_flight

        request_values = np.asarray(self._request_latencies_s, dtype=np.float64)
        return {
            "controller": "bounded_age_double_buffer",
            "control_hz": self.config.control_hz,
            "control_steps": self._control_steps,
            "requests": self._requests,
            "accepted_chunks": self._accepted_chunks,
            "chunk_handoffs": self._chunk_handoffs,
            "hold_steps": self._hold_steps,
            "startup_hold_steps": self._hold_steps - self._buffer_underrun_steps,
            "buffer_underrun_steps": self._buffer_underrun_steps,
            "control_overruns": self._control_overruns,
            "active_remaining": len(self._active_actions),
            "pending_chunk": self._pending_chunk is not None,
            "request_in_flight": in_flight,
            "action_chunk_shape": list(self._chunk_shape or (0, ACTION_DIM)),
            "request_lead_steps": self.config.request_lead_steps,
            "inference_mean_ms": mean_ms,
            "inference_p50_ms": p50_ms,
            "inference_p95_ms": p95_ms,
            "inference_max_ms": max_ms,
            "end_to_end_p95_ms": (
                float(np.percentile(request_values, 95) * 1000) if request_values.size else float("nan")
            ),
            "end_to_end_max_ms": float(np.max(request_values) * 1000) if request_values.size else float("nan"),
        }

    def stop(self) -> None:
        self._stop.set()
        self._request_event.set()

        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=self.config.inference_timeout_s + 2.0)

        if worker is not None and worker.is_alive():
            raise RuntimeError("RTC inference worker did not exit cleanly")

        self._worker = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--preview-port", type=int, default=None,
        help="show the three inference input images at http://127.0.0.1:PORT (e.g. 8765)",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="continuously send camera observations to the model and discard actions without controlling the robot",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="explicit robot configuration directory; use the CR3+O6 config directory",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="exact task prompt used for this checkpoint",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="0 runs until Ctrl-C",
    )
    parser.add_argument(
        "--request-lead-steps",
        type=int,
        default=6,
        help=(
            "request the next chunk when this many actions remain; "
            "6 steps = 300 ms at 20 Hz. Set it to "
            "ceil((observation_ms + inference_ms) * control_hz / 1000) + 1 so the "
            "buffer never underruns (watch 'RTC buffer underrun' in the log)"
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--power-on",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-motion",
        action="store_true",
        help="required before ServoJ motion",
    )

    # Reject obsolete options rather than silently pretending to use them.
    parser.add_argument("--delay-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prefix-horizon", type=int, default=None, help=argparse.SUPPRESS)

    return parser


def _validate_server_metadata(metadata: dict[str, Any], expected_contract: str = ACTION_CONTRACT) -> None:
    for key, expected in (("state_dim", 12), ("physical_action_dim", 12), ("control_hz", POLICY_CONTROL_HZ)):
        if metadata.get(key) != expected:
            raise RuntimeError(f"policy {key} mismatch: expected {expected}, got {metadata.get(key)}")
    contract = metadata.get("action_contract")
    if contract != expected_contract:
        raise RuntimeError(
            "policy action contract mismatch: "
            f"expected {expected_contract!r}, got {contract!r}"
        )


def main() -> None:
    args = _parser().parse_args()

    if args.preview_only:
        if args.preview_port is None:
            raise SystemExit("--preview-only requires --preview-port")
        if args.confirm_motion or args.power_on:
            raise SystemExit("--preview-only cannot be combined with --confirm-motion or --power-on")
    elif not args.confirm_motion:
        raise SystemExit("RTC motion requires --confirm-motion")

    if args.request_lead_steps < 1:
        raise SystemExit("--request-lead-steps must be >= 1")
    if args.preview_port is not None and not 1 <= args.preview_port <= 65535:
        raise SystemExit("--preview-port must be between 1 and 65535")

    if args.delay_steps is not None or args.prefix_horizon is not None:
        raise SystemExit("This client does not implement --delay-steps / --prefix-horizon")

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        inference_timeout_s=args.timeout_s,
    )

    metadata = policy.get_server_metadata()
    validate_task_prompt(metadata, args.prompt)
    interface_config = InterfaceConfig.from_yaml_dir(args.config_dir)
    _validate_server_metadata(metadata, interface_config.action_contract)
    config = RTCDeploymentConfig(
        control_hz=CONTROL_HZ,
        request_lead_steps=args.request_lead_steps,
        inference_timeout_s=args.timeout_s,
        max_observation_age_s=interface_config.action_max_age_s,
    )
    logging.info(
        "RTC timing: policy_hz=%.1f execution_hz=%.1f period_ms=%.3f request_lead_ms=%.1f",
        metadata["control_hz"], config.control_hz, 1000.0 / config.control_hz,
        config.request_lead_steps * 1000.0 / config.control_hz,
    )

    robot = NrcArmO6Interface(
        interface_config
    )

    controller: AsyncDeltaRTCController | None = None
    preview: CameraPreviewPolicy | None = None
    control_started = False

    try:
        if args.preview_port is not None:
            preview = CameraPreviewPolicy(policy, args.preview_port, preview_only=args.preview_only)
            preview.start()
            policy = preview
        robot.connect(start_cameras=True)
        if args.preview_only:
            logging.info("Preview-only mode: model actions are discarded; no robot motion commands will be sent")
            next_inference_at = time.monotonic()
            while True:
                policy.infer(robot.get_observation(args.prompt))
                next_inference_at += PREVIEW_ONLY_PERIOD_S
                delay = next_inference_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_inference_at = time.monotonic()
        else:
            warmup_policy(policy, robot, args.prompt)

            if args.power_on:
                robot.enable_robot()

            robot.start_control()
            control_started = True

            controller = AsyncDeltaRTCController(
                robot=robot,
                policy=policy,
                prompt=args.prompt,
                config=config,
            )

            result = controller.run(
                duration_s=args.duration_s or None
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

    except KeyboardInterrupt:
        logging.info("%s stopped by user", "Input preview" if args.preview_only else "RTC")

    except BaseException:
        if control_started:
            try:
                robot.emergency_stop()
            except Exception:
                logging.exception(
                    "Emergency stop reported an additional error"
                )
        raise

    finally:
        errors = []
        if control_started:
            try:
                robot.stop_control()
            except Exception as exc:
                errors.append(exc)

        try:
            if controller is not None:
                controller.stop()
        except Exception as exc:
            errors.append(exc)
        try:
            robot.close()
        except Exception as exc:
            errors.append(exc)
        if preview is not None:
            try:
                preview.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("RTC shutdown incomplete", errors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
