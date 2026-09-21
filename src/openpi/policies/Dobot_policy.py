"""DOBOT CR5 + LinkerHand O6 collection and OpenPI policy contracts.

This module intentionally keeps the Phase 6 raw-data contract and the Phase 9
training/inference transforms together.  Raw episode recording is not a
LeRobot conversion: it only persists synchronized source observations and the
12D expert command so conversion can be performed in a later phase.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Protocol

import einops
import numpy as np
from PIL import Image

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 12
ACTION_DIM = 12
O6_DIM = 6
RAW_DATA_FPS = 20
DEFAULT_TASK_PROMPT = "pick up the cylinder correctly; if the bump faces down, flip it before grasping"
ACTION_CONTRACT = "tcp_delta_6d_plus_o6_target_6d"
ABSOLUTE_ACTION_CONTRACT = "tcp_absolute_6d_plus_o6_target_6d"
JOINT_ACTION_CONTRACT = "cr3_q1_q6_absolute_plus_o6_target_6d"
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 20


def _finite_vector(values: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


def _finite_last_dim(values: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must end with dimension {size}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


def _parse_image(image: Any) -> np.ndarray:
    """Return one RGB image as contiguous uint8 HWC data."""

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"image must be rank 3, got {array.shape}")
    if array.shape[-1] == 3:
        pass
    elif array.shape[0] == 3:
        array = einops.rearrange(array, "c h w -> h w c")
    else:
        raise ValueError(f"image must be HWC or CHW RGB, got {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("image contains NaN or Inf")
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(f"floating image values must be in [0, 1] or [0, 255], got [{minimum}, {maximum}]")
        if maximum <= 1.0:
            array = array * 255.0
        array = np.rint(array)
    elif not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"unsupported image dtype: {array.dtype}")

    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


class GelloCommandSource(Protocol):
    """Minimal Phase 6 boundary expected from the existing GELLO frontend."""

    def get_command(self) -> Mapping[str, Any]:
        """Return final ``tcp_delta`` and ``o6_target`` values for one cycle."""


class DobotCollectionRobot(Protocol):
    """Robot-side methods used by the 20 Hz collector."""

    def get_images(self) -> Mapping[str, Any]: ...

    def get_state(self) -> np.ndarray: ...

    def apply_action(self, action: Sequence[float]) -> np.ndarray: ...


@dataclasses.dataclass(frozen=True)
class GelloCommandAdapter:
    """Adapt existing GELLO callbacks to the fixed 12D expert-action contract.

    ``read_tcp_delta`` must already apply scale, deadzone and clamp and return
    ``[dX, dY, dZ, dRoll, dPitch, dYaw]`` in metres/radians.  Gesture names are
    resolved to the actual six O6 register targets before the action is logged.
    """

    read_tcp_delta: Callable[[], Sequence[float]]
    read_gesture_id: Callable[[], str]
    gestures: Mapping[str, Sequence[float]]

    def __post_init__(self) -> None:
        if not self.gestures:
            raise ValueError("gestures must not be empty")
        for name, target in self.gestures.items():
            if not str(name).strip():
                raise ValueError("gesture names must not be empty")
            values = _finite_vector(target, O6_DIM, f"gesture {name!r}")
            if np.any((values < 0.0) | (values > 255.0)):
                raise ValueError(f"gesture {name!r} must stay in O6 range 0..255")

    def get_command(self) -> dict[str, np.ndarray]:
        tcp_delta = _finite_vector(self.read_tcp_delta(), 6, "tcp_delta")
        gesture_id = str(self.read_gesture_id()).strip()
        if gesture_id not in self.gestures:
            raise KeyError(f"unknown GELLO gesture: {gesture_id!r}")
        o6_target = _finite_vector(self.gestures[gesture_id], O6_DIM, "o6_target")
        return {"tcp_delta": tcp_delta, "o6_target": o6_target}


def build_expert_action(command: Mapping[str, Any]) -> np.ndarray:
    """Build the exact 12D command prepared for IK and O6 RS485 output."""

    missing = {"tcp_delta", "o6_target"} - set(command)
    if missing:
        raise KeyError(f"GELLO command is missing keys: {sorted(missing)}")
    tcp_delta = _finite_vector(command["tcp_delta"], 6, "tcp_delta")
    o6_target = _finite_vector(command["o6_target"], O6_DIM, "o6_target")
    if np.any((o6_target < 0.0) | (o6_target > 255.0)):
        raise ValueError("o6_target must stay in O6 range 0..255")
    return np.concatenate([tcp_delta, o6_target]).astype(np.float32)


class RawEpisodeRecorder:
    """Persist one Phase 6 episode without performing any format conversion."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        episode_index: int,
        task: str = DEFAULT_TASK_PROMPT,
        orientation: str,
        fps: int = RAW_DATA_FPS,
    ) -> None:
        if int(episode_index) < 0:
            raise ValueError("episode_index must be non-negative")
        if int(fps) != RAW_DATA_FPS:
            raise ValueError(f"Phase 6 collection must run at {RAW_DATA_FPS} Hz")
        if not str(task).strip():
            raise ValueError("task must not be empty")
        if not str(orientation).strip():
            raise ValueError("orientation must not be empty")

        self.fps = int(fps)
        self.task = str(task).strip()
        self.orientation = str(orientation).strip()
        self.episode_dir = Path(root_dir).expanduser().resolve() / f"episode_{int(episode_index):06d}"
        self.global_dir = self.episode_dir / "global"
        self.wrist_dir = self.episode_dir / "wrist"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.global_dir.mkdir()
        self.wrist_dir.mkdir()

        self._timestamp_ns: list[int] = []
        self._global_timestamp_ns: list[int] = []
        self._wrist_timestamp_ns: list[int] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._executed_action: list[np.ndarray] = []
        self._overrun_count = 0
        self._finalized = False

    @property
    def sample_count(self) -> int:
        return len(self._timestamp_ns)

    def write(
        self,
        *,
        timestamp_ns: int,
        global_rgb: Any,
        wrist_rgb: Any,
        state: Any,
        action: Any,
        global_timestamp_ns: int = -1,
        wrist_timestamp_ns: int = -1,
    ) -> int:
        """Write ``(observation_t, expert_action_t)`` before robot execution."""

        if self._finalized:
            raise RuntimeError("episode has already been finalized")
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if self._timestamp_ns and timestamp_ns <= self._timestamp_ns[-1]:
            raise ValueError("sample timestamps must be strictly increasing")

        state_array = _finite_vector(state, STATE_DIM, "state")
        action_array = _finite_vector(action, ACTION_DIM, "action")
        global_image = _parse_image(global_rgb)
        wrist_image = _parse_image(wrist_rgb)
        index = self.sample_count

        Image.fromarray(global_image).save(self.global_dir / f"{index:06d}.jpg", quality=95, subsampling=0)
        Image.fromarray(wrist_image).save(self.wrist_dir / f"{index:06d}.jpg", quality=95, subsampling=0)

        self._timestamp_ns.append(timestamp_ns)
        self._global_timestamp_ns.append(int(global_timestamp_ns))
        self._wrist_timestamp_ns.append(int(wrist_timestamp_ns))
        self._state.append(state_array)
        self._action.append(action_array)
        self._executed_action.append(np.full(ACTION_DIM, np.nan, dtype=np.float32))
        return index

    def confirm_execution(self, sample_index: int, executed_action: Any) -> None:
        """Attach the safety-filtered command returned by the robot interface."""

        if int(sample_index) != self.sample_count - 1:
            raise ValueError("only the latest sample can be confirmed")
        if np.all(np.isfinite(self._executed_action[sample_index])):
            raise RuntimeError(f"sample {sample_index} execution is already confirmed")
        self._executed_action[sample_index] = _finite_vector(executed_action, ACTION_DIM, "executed_action")

    def note_overrun(self) -> None:
        self._overrun_count += 1

    def finalize(self) -> Path:
        if self._finalized:
            return self.episode_dir
        if not self._timestamp_ns:
            raise RuntimeError("cannot finalize an empty episode")
        if not all(np.all(np.isfinite(action)) for action in self._executed_action):
            raise RuntimeError("cannot finalize: at least one recorded action was not executed successfully")

        records_path = self.episode_dir / "records.npz"
        records_tmp = self.episode_dir / "records.npz.tmp"
        with records_tmp.open("wb") as stream:
            np.savez_compressed(
                stream,
                timestamp_ns=np.asarray(self._timestamp_ns, dtype=np.int64),
                global_timestamp_ns=np.asarray(self._global_timestamp_ns, dtype=np.int64),
                wrist_timestamp_ns=np.asarray(self._wrist_timestamp_ns, dtype=np.int64),
                state=np.stack(self._state).astype(np.float32),
                action=np.stack(self._action).astype(np.float32),
                executed_action=np.stack(self._executed_action).astype(np.float32),
            )
        records_tmp.replace(records_path)

        metadata = {
            "fps": self.fps,
            "task": self.task,
            "orientation": self.orientation,
            "action_contract": ACTION_CONTRACT,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "sample_count": self.sample_count,
            "overrun_count": self._overrun_count,
            "executed_action_is_safety_filtered": True,
        }
        metadata_path = self.episode_dir / "metadata.json"
        metadata_tmp = self.episode_dir / "metadata.json.tmp"
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata_tmp.replace(metadata_path)
        self._finalized = True
        return self.episode_dir


def collect_expert_episode(
    robot: DobotCollectionRobot,
    gello: GelloCommandSource,
    recorder: RawEpisodeRecorder,
    *,
    num_steps: int | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> Path:
    """Collect one 20 Hz episode; robot connection/control stays caller-owned.

    The observation and proposed expert action are recorded before execution.
    The action returned by ``robot.apply_action`` is stored separately as the
    exact safety-filtered command, preserving the distinction without using a
    future state as the training label.
    """

    if recorder.fps != RAW_DATA_FPS:
        raise ValueError(f"recorder must use {RAW_DATA_FPS} Hz")
    if num_steps is None and stop_requested is None:
        raise ValueError("provide num_steps or stop_requested")
    if num_steps is not None and int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")

    period_s = 1.0 / RAW_DATA_FPS
    next_cycle = time.monotonic()
    completed = 0
    while num_steps is None or completed < int(num_steps):
        if stop_requested is not None and stop_requested():
            break
        timestamp_ns = time.monotonic_ns()
        images = robot.get_images()
        missing_images = {"global_rgb", "wrist_rgb"} - set(images)
        if missing_images:
            raise KeyError(f"robot images are missing keys: {sorted(missing_images)}")
        state = robot.get_state()
        action = build_expert_action(gello.get_command())

        sample_index = recorder.write(
            timestamp_ns=timestamp_ns,
            global_rgb=images["global_rgb"],
            wrist_rgb=images["wrist_rgb"],
            state=state,
            action=action,
            global_timestamp_ns=int(images.get("global_timestamp_ns", -1)),
            wrist_timestamp_ns=int(images.get("wrist_timestamp_ns", -1)),
        )
        executed_action = robot.apply_action(action)
        recorder.confirm_execution(sample_index, executed_action)
        completed += 1

        next_cycle += period_s
        delay = next_cycle - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            recorder.note_overrun()
            next_cycle = time.monotonic()

    if completed == 0:
        raise RuntimeError("episode stopped before the first sample")
    return recorder.finalize()


def make_dobot_cr5_o6_example() -> dict[str, Any]:
    """Create one inference-shaped example for transform smoke tests."""

    return {
        "observation/global_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/wrist_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/state": np.zeros(STATE_DIM, dtype=np.float32),
        "prompt": DEFAULT_TASK_PROMPT,
    }


@dataclasses.dataclass(frozen=True)
class DobotCR5O6Inputs(transforms.DataTransformFn):
    """Map physical CR5/O6 observations to the three-image OpenPI contract."""

    model_type: _model.ModelType

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"DobotCR5O6Inputs does not support {self.model_type}")

        global_rgb = _parse_image(data["observation/global_rgb"])
        wrist_rgb = _parse_image(data["observation/wrist_rgb"])
        right_wrist_rgb = _parse_image(data["observation/right_wrist_rgb"])
        state = _finite_last_dim(data["observation/state"], STATE_DIM, "observation/state")

        # CR5 joint/TCP values keep their physical units; O6 feedback becomes 0..1.
        state[..., 6:12] /= 255.0
        inputs: dict[str, Any] = {
            "state": state,
            "image": {
                "base_0_rgb": global_rgb,
                "left_wrist_0_rgb": wrist_rgb,
                "right_wrist_0_rgb": right_wrist_rgb,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = _finite_last_dim(data["actions"], ACTION_DIM, "actions")
            # TCP values already use the configured physical representation
            # (residual or absolute). Only the O6 register units are scaled here.
            actions[..., 6:12] /= 255.0
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


def _joint_image(image: Any) -> np.ndarray:
    """Match the training conversion exactly: direct 224x224 resize, no padding.

    ``examples/cr3_o6/convert_data_to_lerobot.py`` writes every training frame
    with ``frame.reformat(width=224, height=224)``, i.e. a full-frame anamorphic
    resize of the 640x480 session video (verified: those dataset videos contain
    0 black pixels). Serving must use the same geometry, otherwise the policy
    sees a letterboxed 224x168 view with 25% black bars that it never saw in
    training.

    Datasets collected with a padded 224x168-in-224x224 geometry require a
    matching transform. This baseline uses direct resize because its conversion
    path writes full-frame 224x224 images.
    """
    import cv2

    image = _parse_image(image)
    if image.shape[:2] == (224, 224):
        return image
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


@dataclasses.dataclass(frozen=True)
class CR3O6JointInputs(transforms.DataTransformFn):
    """Absolute joint positions and O6 registers; no TCP or delta transform."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = _finite_last_dim(data["observation/state"], 12, "observation/state")
        state[..., 6:12] /= 255.0
        result = {
            "state": state,
            "image": {
                "base_0_rgb": _joint_image(data["observation/global_rgb"]),
                "left_wrist_0_rgb": _joint_image(data["observation/wrist_rgb"]),
                "right_wrist_0_rgb": _joint_image(data["observation/right_wrist_rgb"]),
            },
            "image_mask": {key: np.True_ for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        }
        if "actions" in data:
            actions = _finite_last_dim(data["actions"], 12, "actions")
            actions[..., 6:12] /= 255.0
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class DobotCR5O6Outputs(transforms.DataTransformFn):
    """Convert the first 12 model outputs back to the physical action contract."""

    def __call__(self, data: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 0 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"model actions must provide at least {ACTION_DIM} values, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("model actions contain NaN or Inf")
        actions = actions[..., :ACTION_DIM].copy()
        actions[..., 6:12] = np.clip(actions[..., 6:12] * 255.0, 0.0, 255.0)
        return {"actions": actions}


class PolicyClient(Protocol):
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]: ...


class DobotDeploymentRobot(Protocol):
    def get_observation(self, prompt: str) -> dict[str, Any]: ...

    def apply_action(self, action: Sequence[float]) -> np.ndarray: ...

    def hold_action(self) -> np.ndarray: ...


@dataclasses.dataclass(frozen=True)
class RTCDeploymentConfig:
    """Fixed-shape Phase 13-16 RTC deployment contract."""

    control_hz: float = 20.0
    action_horizon: int = ACTION_HORIZON
    physical_action_dim: int = ACTION_DIM
    model_action_dim: int = MODEL_ACTION_DIM
    initial_inference_delay_steps: int = 4
    prefix_attention_horizon: int = 10
    num_denoise_steps: int = 10
    max_guidance_weight: float = 10.0
    inference_timeout_s: float = 5.0
    max_consecutive_inference_errors: int = 3
    request_poll_s: float = 0.005
    latency_history_size: int = 100

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.physical_action_dim != ACTION_DIM:
            raise ValueError(f"physical_action_dim must stay {ACTION_DIM}")
        if self.model_action_dim < self.physical_action_dim:
            raise ValueError("model_action_dim cannot be smaller than physical_action_dim")
        if not 0 <= self.initial_inference_delay_steps < self.action_horizon:
            raise ValueError("initial_inference_delay_steps must be inside the action horizon")
        if not 1 <= self.prefix_attention_horizon <= self.action_horizon:
            raise ValueError("prefix_attention_horizon must be in [1, action_horizon]")
        if self.num_denoise_steps <= 0:
            raise ValueError("num_denoise_steps must be positive")
        if not np.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be finite and positive")
        if self.inference_timeout_s <= 0 or self.request_poll_s <= 0:
            raise ValueError("RTC timeout and poll interval must be positive")
        if self.max_consecutive_inference_errors <= 0 or self.latency_history_size <= 0:
            raise ValueError("RTC error/history limits must be positive")

    @classmethod
    def from_policy_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
        **overrides: Any,
    ) -> RTCDeploymentConfig:
        raw: Mapping[str, Any] = {}
        if metadata is not None:
            candidate = metadata.get("rtc", {})
            if not isinstance(candidate, Mapping):
                raise TypeError("policy metadata field 'rtc' must be a mapping")
            raw = candidate
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted((set(raw) | set(overrides)) - allowed)
        if unknown:
            raise ValueError(f"unknown RTC configuration fields: {unknown}")
        values = dict(raw)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def policy_request(self, previous: np.ndarray, inference_delay: int) -> dict[str, Any]:
        return {
            "prev_action_chunk": previous,
            "inference_delay": int(inference_delay),
            "prefix_attention_horizon": self.prefix_attention_horizon,
            "num_steps": self.num_denoise_steps,
            "real_action_dim": self.physical_action_dim,
            "max_guidance_weight": self.max_guidance_weight,
        }


class RTCBroker:
    """Thread-safe fixed-horizon broker for one inference/control pair."""

    def __init__(self, config: RTCDeploymentConfig):
        self.config = config
        self.physical_chunk: np.ndarray | None = None
        self.model_chunk: np.ndarray | None = None
        self.cursor = 0
        self.request_step: int | None = None
        self.inference_running = False
        self.latency_history: list[float] = []
        self.global_step = 0
        self.discarded_responses = 0
        self.last_error = ""
        self.control_overruns = 0
        self._request_started_s: float | None = None
        self._lock = threading.RLock()

    def _remaining_unlocked(self) -> int:
        if self.physical_chunk is None:
            return 0
        return max(0, self.config.action_horizon - self.cursor)

    def estimated_delay_steps(self) -> int:
        with self._lock:
            if self.latency_history:
                seconds = float(np.percentile(np.asarray(self.latency_history), 95))
                delay = math.ceil(seconds * self.config.control_hz)
            else:
                delay = self.config.initial_inference_delay_steps
            return int(np.clip(delay, 0, self.config.action_horizon - 1))

    def should_request(self) -> bool:
        with self._lock:
            if self.inference_running:
                return False
            lead = self.estimated_delay_steps() + self.config.prefix_attention_horizon
            return self.physical_chunk is None or self._remaining_unlocked() <= lead

    def _aligned_previous_unlocked(self) -> np.ndarray | None:
        if self.model_chunk is None or self.cursor >= self.config.action_horizon:
            return None
        left = self.model_chunk[self.cursor :]
        padding = np.zeros((self.cursor, self.config.model_action_dim), dtype=np.float32)
        aligned = np.concatenate([left, padding], axis=0)
        expected = (self.config.action_horizon, self.config.model_action_dim)
        if aligned.shape != expected:
            raise RuntimeError(f"aligned previous model chunk must be {expected}, got {aligned.shape}")
        return aligned

    def begin_request(self) -> dict[str, Any]:
        with self._lock:
            if self.inference_running:
                raise RuntimeError("an RTC inference request is already running")
            self.inference_running = True
            self.request_step = self.global_step
            self._request_started_s = time.monotonic()
            return {
                "request_step": self.request_step,
                "previous": self._aligned_previous_unlocked(),
                "inference_delay": self.estimated_delay_steps(),
            }

    def complete_request(
        self,
        result: Mapping[str, Any],
        *,
        request_step: int,
        latency_s: float,
    ) -> bool:
        physical = _finite_last_dim(result["actions"], self.config.physical_action_dim, "actions")
        model = _finite_last_dim(result["actions_model"], self.config.model_action_dim, "actions_model")
        expected_physical = (self.config.action_horizon, self.config.physical_action_dim)
        expected_model = (self.config.action_horizon, self.config.model_action_dim)
        if physical.shape != expected_physical:
            raise ValueError(f"physical chunk must be {expected_physical}, got {physical.shape}")
        if model.shape != expected_model:
            raise ValueError(f"model chunk must be {expected_model}, got {model.shape}")
        if not np.isfinite(latency_s) or latency_s < 0:
            raise ValueError("latency_s must be finite and non-negative")

        with self._lock:
            if not self.inference_running or self.request_step != int(request_step):
                raise RuntimeError("RTC response does not match the active request")
            actual_delay = self.global_step - int(request_step)
            self.inference_running = False
            self.request_step = None
            self._request_started_s = None
            self.latency_history.append(float(latency_s))
            del self.latency_history[: -self.config.latency_history_size]
            if actual_delay >= self.config.action_horizon:
                self.discarded_responses += 1
                return False

            # Keep the full H-step response and start at actual_delay. This is
            # equivalent to installing result[actual_delay:], while preserving
            # an H x model_action_dim previous chunk for the next JIT request.
            self.physical_chunk = physical
            self.model_chunk = model
            self.cursor = actual_delay
            self.last_error = ""
            return True

    def fail_request(self, request_step: int, error: BaseException) -> None:
        with self._lock:
            if self.request_step == int(request_step):
                self.inference_running = False
                self.request_step = None
                self._request_started_s = None
            self.last_error = f"{type(error).__name__}: {error}"

    def next_control_action(self) -> np.ndarray | None:
        """Consume one tick atomically; ``None`` means use zero-delta hold."""

        with self._lock:
            action: np.ndarray | None = None
            if self.physical_chunk is not None and self.cursor < self.config.action_horizon:
                action = self.physical_chunk[self.cursor].copy()
                self.cursor += 1
            self.global_step += 1
            return action

    def note_control_overrun(self) -> None:
        with self._lock:
            self.control_overruns += 1

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "global_step": self.global_step,
                "cursor": self.cursor,
                "remaining_actions": self._remaining_unlocked(),
                "inference_running": self.inference_running,
                "estimated_delay_steps": self.estimated_delay_steps(),
                "latency_samples": len(self.latency_history),
                "discarded_responses": self.discarded_responses,
                "control_overruns": self.control_overruns,
                "last_error": self.last_error,
            }


class AsyncRTCController:
    """Run policy inference asynchronously while dispatching actions at 20 Hz."""

    def __init__(
        self,
        robot: DobotDeploymentRobot,
        policy: PolicyClient,
        prompt: str,
        config: RTCDeploymentConfig,
    ) -> None:
        if not str(prompt).strip():
            raise ValueError("prompt must not be empty")
        self.robot = robot
        self.policy = policy
        self.prompt = str(prompt).strip()
        self.config = config
        self.broker = RTCBroker(config)
        self._stop = threading.Event()
        self._inference_thread: threading.Thread | None = None
        self._fatal_error: BaseException | None = None

    def _inference_loop(self) -> None:
        consecutive_errors = 0
        while not self._stop.is_set():
            if not self.broker.should_request():
                self._stop.wait(self.config.request_poll_s)
                continue

            request = self.broker.begin_request()
            request_step = int(request["request_step"])
            started = time.monotonic()
            try:
                observation = self.robot.get_observation(self.prompt)
                previous = request["previous"]
                if previous is not None:
                    observation["_rtc"] = self.config.policy_request(
                        previous,
                        int(request["inference_delay"]),
                    )
                result = self.policy.infer(observation)
                self.broker.complete_request(
                    result,
                    request_step=request_step,
                    latency_s=time.monotonic() - started,
                )
                consecutive_errors = 0
            except BaseException as exc:
                self.broker.fail_request(request_step, exc)
                consecutive_errors += 1
                if consecutive_errors >= self.config.max_consecutive_inference_errors:
                    self._fatal_error = RuntimeError(
                        f"RTC inference failed {consecutive_errors} consecutive times: {exc}"
                    )
                    self._stop.set()
                else:
                    self._stop.wait(min(0.1, self.config.request_poll_s * 10))

    def start(self) -> None:
        if self._inference_thread is not None:
            raise RuntimeError("RTC controller has already been started")
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="dobot-rtc-inference",
            daemon=True,
        )
        self._inference_thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._inference_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.config.inference_timeout_s + 0.5)

    def run(self, *, duration_s: float | None = None) -> dict[str, Any]:
        """Block in the control loop; robot lifecycle remains caller-owned."""

        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive when provided")
        self.start()
        period_s = 1.0 / self.config.control_hz
        started = time.monotonic()
        next_cycle = started
        try:
            while not self._stop.is_set():
                if self._fatal_error is not None:
                    raise self._fatal_error
                if duration_s is not None and time.monotonic() - started >= duration_s:
                    break

                action = self.broker.next_control_action()
                if action is None:
                    # Never repeat a stale TCP delta while inference is late.
                    action = _finite_vector(self.robot.hold_action(), ACTION_DIM, "hold_action")
                    action[:6] = 0.0
                self.robot.apply_action(action)

                next_cycle += period_s
                delay = next_cycle - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    self.broker.note_control_overrun()
                    next_cycle = time.monotonic()
        finally:
            self.stop()
        if self._fatal_error is not None:
            raise self._fatal_error
        return self.broker.diagnostics()


def benchmark_policy_latency(
    policy: PolicyClient,
    observation: Mapping[str, Any],
    *,
    samples: int = 100,
    control_hz: float = RAW_DATA_FPS,
) -> dict[str, float | int]:
    """Measure full client/server RTT without executing a robot action."""

    if samples <= 0 or not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("samples and control_hz must be positive")
    latencies = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        started = time.monotonic()
        policy.infer(dict(observation))
        latencies[index] = time.monotonic() - started
    return {
        "samples": samples,
        "p50_ms": float(np.percentile(latencies, 50) * 1000),
        "p90_ms": float(np.percentile(latencies, 90) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "p99_ms": float(np.percentile(latencies, 99) * 1000),
        "recommended_delay_steps": math.ceil(float(np.percentile(latencies, 95)) * control_hz),
    }


__all__ = [
    "ACTION_CONTRACT",
    "ACTION_DIM",
    "ACTION_HORIZON",
    "DEFAULT_TASK_PROMPT",
    "MODEL_ACTION_DIM",
    "RAW_DATA_FPS",
    "STATE_DIM",
    "AsyncRTCController",
    "DobotCR5O6Inputs",
    "DobotCR5O6Outputs",
    "GelloCommandAdapter",
    "RTCBroker",
    "RTCDeploymentConfig",
    "RawEpisodeRecorder",
    "benchmark_policy_latency",
    "build_expert_action",
    "collect_expert_episode",
    "make_dobot_cr5_o6_example",
]
