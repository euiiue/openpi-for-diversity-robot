from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import main_rtc
from main_rtc import AsyncDeltaRTCController
from main_rtc import RTCDeploymentConfig
from main_rtc import _InferenceResult


@pytest.mark.parametrize("observation_ms", [0, 40, 80])
def test_rtc_boundary_arrival_vs_real_buffer_underrun(monkeypatch, caplog, observation_ms):
    """Run the real control loop with a deterministic clock and queued worker results."""
    now_ns = [100_000_000_000]
    monkeypatch.setattr(main_rtc.time, "monotonic_ns", lambda: now_ns[0])
    monkeypatch.setattr(main_rtc.time, "monotonic", lambda: now_ns[0] / 1e9)
    monkeypatch.setattr(main_rtc.threading, "Thread", Mock())
    robot = Mock()
    controller = AsyncDeltaRTCController(
        robot, Mock(), "offline RTC test", RTCDeploymentConfig(control_hz=20, request_lead_steps=6)
    )
    total_latency_ns = (observation_ms + 250) * 1_000_000

    def advance_tick(delay_s):
        now_ns[0] += round(delay_s * 1e9)
        ready_ns = controller._request_started_ns + total_latency_ns
        if controller._request_in_flight and controller._results.empty() and now_ns[0] >= ready_ns:
            # Only the asynchronous worker is simulated. The actual run(),
            # request scheduler, queue consumption, activation and age checks run.
            controller._results.put(
                _InferenceResult(
                    request_id=controller._requests,
                    actions=np.full((20, 12), controller._requests, dtype=np.float32),
                    latency_s=0.25,
                    observation_timestamp_ns=controller._request_started_ns,
                    request_started_ns=controller._request_started_ns,
                    observation_latency_s=observation_ms / 1000,
                    ready_ns=ready_ns,
                )
            )

    monkeypatch.setattr(controller._stop, "wait", advance_tick)
    with caplog.at_level("INFO"):
        result = controller.run(duration_s=4)

    assert result["control_steps"] == 80
    assert result["control_overruns"] == 0
    assert result["chunk_handoffs"] >= 3
    assert result["startup_hold_steps"] == int(np.ceil((observation_ms + 250) / 50))
    assert result["end_to_end_max_ms"] == int(np.ceil((observation_ms + 250) / 50)) * 50
    assert result["inference_max_ms"] == 250
    assert "active_remaining=0" in caplog.text
    assert f"request_to_ready_ms={observation_ms + 250:.1f}" in caplog.text
    if observation_ms <= 40:
        # active_remaining=0 at receipt can still have ZERO steady-state holds.
        assert result["buffer_underrun_steps"] == 0
        assert "RTC buffer underrun:" not in caplog.text
    else:
        assert result["buffer_underrun_steps"] > 0
        assert "RTC buffer underrun:" in caplog.text
    assert result["hold_steps"] == result["startup_hold_steps"] + result["buffer_underrun_steps"]
    assert robot.track_or_hold.call_count == result["hold_steps"]
    sent = [call.args[0][0] for call in robot.apply_action.call_args_list]
    np.testing.assert_array_equal(sent, np.repeat(np.arange(1, 5), 20)[:len(sent)])


def test_pending_chunk_does_not_refresh_observation_timestamp(monkeypatch):
    monkeypatch.setattr(main_rtc.time, "monotonic_ns", lambda: 100_400_000_000)
    controller = AsyncDeltaRTCController(
        Mock(), Mock(), "offline RTC test", RTCDeploymentConfig(request_lead_steps=8)
    )
    controller._pending_chunk = _InferenceResult(
        request_id=1,
        actions=np.zeros((20, 12)),
        latency_s=0.25,
        observation_timestamp_ns=100_000_000_000,
        request_started_ns=100_000_000_000,
        observation_latency_s=0,
        ready_ns=100_250_000_000,
    )
    with pytest.raises(TimeoutError, match="age invalid"):
        controller._promote_pending_if_boundary()
    assert not controller._active_actions
    controller.robot.apply_action.assert_not_called()


def test_20_hz_executes_all_20_rows_with_retimed_timestamps(monkeypatch, caplog):
    now_ns = [100_000_000_000]
    monkeypatch.setattr(main_rtc.time, "monotonic_ns", lambda: now_ns[0])
    monkeypatch.setattr(main_rtc.time, "monotonic", lambda: now_ns[0] / 1e9)
    monkeypatch.setattr(main_rtc.threading, "Thread", Mock())
    robot = Mock()
    controller = AsyncDeltaRTCController(
        robot, Mock(), "offline 20 Hz test", RTCDeploymentConfig(control_hz=20, request_lead_steps=6)
    )
    sent_at_ns = []
    robot.apply_action.side_effect = lambda *args, **kwargs: sent_at_ns.append(now_ns[0])

    def advance_tick(delay_s):
        now_ns[0] += round(delay_s * 1e9)
        ready_ns = controller._request_started_ns + 10_000_000
        if controller._request_in_flight and controller._results.empty() and now_ns[0] >= ready_ns:
            rows = np.arange(20, dtype=np.float32) + controller._requests * 20
            controller._results.put(
                _InferenceResult(
                    request_id=controller._requests,
                    actions=np.repeat(rows[:, None], 12, axis=1),
                    latency_s=0.01,
                    observation_timestamp_ns=controller._request_started_ns,
                    request_started_ns=controller._request_started_ns,
                    observation_latency_s=0,
                    ready_ns=ready_ns,
                )
            )

    monkeypatch.setattr(controller._stop, "wait", advance_tick)
    with caplog.at_level("INFO"):
        # Half-tick end margin avoids floating-point equality at the deadline.
        result = controller.run(duration_s=40.5 / 20)

    assert result["control_hz"] == 20
    assert result["control_steps"] == 41
    assert result["startup_hold_steps"] == 1
    assert result["buffer_underrun_steps"] == 0
    # The loop activates the next pending chunk before checking duration, but
    # must send no rows from that third chunk after the duration limit.
    assert result["chunk_handoffs"] == 3
    assert result["active_remaining"] == 20
    assert result["action_chunk_shape"] == [20, 12]
    sent = robot.apply_action.call_args_list
    np.testing.assert_array_equal([call.args[0][0] for call in sent], np.arange(20, 60))
    np.testing.assert_allclose(np.diff(sent_at_ns), 1e9 / 20, rtol=0, atol=1)
    stamps = [call.kwargs["observation_timestamp_ns"] for call in sent]
    expected_offsets = [round(index * 1e9 / 20) for index in range(20)]
    for chunk_start in (0, 20):
        offsets = [stamp - stamps[chunk_start] for stamp in stamps[chunk_start:chunk_start + 20]]
        np.testing.assert_array_equal(offsets, expected_offsets)
    assert "nominal_lead_ms=300.0" in caplog.text
    assert "nominal_chunk_ms=1000.0" in caplog.text


def test_20_hz_deployment_checks_checkpoint_20_hz_contract():
    metadata = {"state_dim": 12, "physical_action_dim": 12, "control_hz": 20,
                "action_contract": main_rtc.ACTION_CONTRACT}
    main_rtc._validate_server_metadata(metadata)
    with pytest.raises(RuntimeError, match="control_hz mismatch"):
        main_rtc._validate_server_metadata({**metadata, "control_hz": 30})
