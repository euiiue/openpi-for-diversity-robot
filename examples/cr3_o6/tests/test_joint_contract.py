from pathlib import Path
import sys
import time
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from interface import InterfaceConfig
from interface import NrcArmO6Interface

from openpi import transforms
from openpi.policies.Dobot_policy import CR3O6JointInputs
from openpi.policies.Dobot_policy import DobotCR5O6Outputs
from openpi.training import config


@pytest.fixture
def robot():
    device = NrcArmO6Interface(
        InterfaceConfig(
            workspace_min_m=(-1, -1, -1), workspace_max_m=(1, 1, 1), max_joint_step_rad=0.12, max_tracking_error_rad=0.1
        )
    )
    device.cr5 = Mock()
    device.cr5.connections_ready.return_value = True
    device.cr5.servo_state.return_value = 3
    device.cr5.joint_position.return_value = [0, 0, 0, 0, 0, 0, 17]
    device.cr5.tcp_position.return_value = [0] * 7
    device.cr5.forward_kinematics.return_value = np.zeros(7)
    device.o6 = Mock(error="")
    device.o6.get_positions.return_value = np.full(6, 100, dtype=np.float32)
    device.o6.set_positions.side_effect = lambda values: np.rint(values)
    device._control_started = True
    device._last_joint_command_time = time.monotonic() - 0.05
    device._last_joint_target_deg = np.array([0, 0, 0, 0, 0, 0, 17], dtype=np.float32)
    return device


def test_joint_action_goes_directly_to_servoj(robot):
    action = np.array([0.01, -0.02, 0.003, -0.004, 0.005, -0.006, 10, 20, 30, 40, 50, 60])
    accepted = robot.apply_action(action)
    sent = robot.cr5.send_servoj.call_args.args[0]
    np.testing.assert_allclose(sent[:6], np.rad2deg(action[:6]), rtol=1e-6)
    assert sent[6] == 17
    robot.cr5.inverse_kinematics.assert_not_called()
    np.testing.assert_allclose(accepted, action, rtol=1e-6)
    np.testing.assert_allclose(robot.hold_action()[:6], action[:6], rtol=1e-6)


@pytest.mark.parametrize("action", [np.r_[np.zeros(6), np.full(6, 256)], np.full(12, np.nan)])
def test_invalid_joint_action_stops_before_send(robot, action):
    with pytest.raises((ValueError, RuntimeError)):
        robot.apply_action(action)
    robot.cr5.send_servoj.assert_not_called()
    robot.cr5.stop_motion.assert_called_once()


def test_previous_target_tracking_error_is_not_overwritten(robot):
    robot._last_joint_target_deg[:6] = 6.5
    with pytest.raises(RuntimeError, match="跟踪误差"):
        robot.apply_action(np.r_[np.zeros(6), np.ones(6)])
    robot.cr5.send_servoj.assert_not_called()


def test_dataset_repack_and_physical_roundtrip():
    cfg = config.get_config("pi05_cr3_o6_joint_abs_lora")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data_cfg.action_sequence_keys == ("action",)
    assert data_cfg.prompt_from_task
    state = np.arange(12, dtype=np.float32)
    action = np.tile(state, (20, 1))
    sample = {"observation.state": state, "action": action, "prompt": "test task"}
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        sample[f"observation.images.{key}"] = np.zeros((3, 224, 224), np.float32)
    repacked = transforms.compose(data_cfg.repack_transforms.inputs)(sample)
    result = CR3O6JointInputs()(repacked)
    np.testing.assert_array_equal(result["state"][:6], state[:6])
    np.testing.assert_allclose(result["state"][6:], state[6:] / 255)
    np.testing.assert_allclose(DobotCR5O6Outputs()(result)["actions"], action)
    np.testing.assert_array_equal(sample["action"], action)
    assert result["prompt"] == "test task"


def test_joint_inputs_accept_full_right_wrist_camera():
    right_wrist_rgb = np.full((480, 640, 3), 37, dtype=np.uint8)
    observation = {
        "observation/global_rgb": np.zeros_like(right_wrist_rgb),
        "observation/wrist_rgb": np.zeros_like(right_wrist_rgb),
        "observation/right_wrist_rgb": right_wrist_rgb,
        "observation/state": np.zeros(12, dtype=np.float32),
        "prompt": "test task",
    }

    result = CR3O6JointInputs()(observation)

    transformed = result["image"]["right_wrist_0_rgb"]
    assert transformed.shape == (224, 224, 3)
    np.testing.assert_array_equal(transformed, 37)


def test_shipped_config_is_joint():
    config = InterfaceConfig.from_yaml_dir()
    assert config.action_contract == NrcArmO6Interface.ACTION_CONTRACT


def test_local_config_overrides_its_example(tmp_path):
    (tmp_path / "robot.example.yaml").write_text(
        "controller_ip: 192.168.1.10\no6_port: /dev/ttyUSB0\n",
        encoding="utf-8",
    )
    (tmp_path / "camera.example.yaml").write_text(
        "global_camera_serial: null\nwrist_camera_serial: null\nright_wrist_camera_serial: null\n",
        encoding="utf-8",
    )
    (tmp_path / "robot.local.yaml").write_text(
        "controller_ip: 10.0.0.44\n",
        encoding="utf-8",
    )

    result = InterfaceConfig.from_yaml_dir(tmp_path)

    assert result.controller_ip == "10.0.0.44"
    assert result.o6_port == "/dev/ttyUSB0"


def test_local_config_rejects_unknown_key(tmp_path):
    (tmp_path / "robot.example.yaml").write_text(
        "controller_ip: 192.168.1.10\no6_port: /dev/ttyUSB0\n",
        encoding="utf-8",
    )
    (tmp_path / "camera.example.yaml").write_text("", encoding="utf-8")
    (tmp_path / "camera.local.yaml").write_text(
        "camera_seral: wrong\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="camera.local.yaml.*未知配置项"):
        InterfaceConfig.from_yaml_dir(tmp_path)


def test_example_config_blocks_motion_without_local_settings(tmp_path):
    (tmp_path / "robot.example.yaml").write_text(
        "controller_ip: 192.168.1.10\no6_port: /dev/ttyUSB0\n",
        encoding="utf-8",
    )
    (tmp_path / "camera.example.yaml").write_text("", encoding="utf-8")

    robot = NrcArmO6Interface(InterfaceConfig.from_yaml_dir(tmp_path))

    with pytest.raises(RuntimeError, match="max_joint_step_rad"):
        robot.start_control()


def test_get_images_reads_three_independent_camera_frames():
    now = time.monotonic_ns()
    frames = {
        "global_rgb": np.full((480, 640, 3), 20, dtype=np.uint8),
        "wrist_rgb": np.full((480, 640, 3), 80, dtype=np.uint8),
        "right_wrist_rgb": np.full((480, 640, 3), 160, dtype=np.uint8),
    }
    config = InterfaceConfig(
        global_camera_serial="base",
        wrist_camera_serial="left",
        right_wrist_camera_serial="right",
    )
    robot = NrcArmO6Interface(config)
    robot._global_camera = Mock()
    robot._wrist_camera = Mock()
    robot._right_wrist_camera = Mock()
    for offset, key in enumerate(frames):
        camera = {
            "global_rgb": robot._global_camera,
            "wrist_rgb": robot._wrist_camera,
            "right_wrist_rgb": robot._right_wrist_camera,
        }[key]
        camera.latest.return_value = (frames[key], now + offset * 1_000_000)
    robot.get_state = Mock(return_value=np.zeros(12, dtype=np.float32))

    observation = robot.get_observation("test task")

    for key, frame in frames.items():
        np.testing.assert_array_equal(observation[f"observation/{key}"], frame)
        assert observation[f"observation/{key}"].shape == (480, 640, 3)
    assert observation["right_wrist_timestamp_ns"] == now + 2_000_000


def test_joint_target_workspace_rejected(robot):
    robot.cr5.forward_kinematics.return_value = np.array([2000, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="workspace") as error:
        robot.apply_action(np.r_[np.zeros(6), np.ones(6)])
    assert "candidate_tcp_mm=[2000.0, 0.0, 0.0]" in str(error.value)
    assert "feedback_tcp_mm=[0.0, 0.0, 0.0]" in str(error.value)
    assert "command_min_mm=[-1000, -1000, -1000]" in str(error.value)
    assert "command_max_mm=[1000, 1000, 1000]" in str(error.value)
    assert "outside_axes=['x']" in str(error.value)
    assert "excess_mm=[1000.0, 0.0, 0.0]" in str(error.value)
    robot.cr5.send_servoj.assert_not_called()
    robot.cr5.stop_motion.assert_called_once()


def test_home_endpoint_is_preserved_and_rate_limited(robot):
    goal = np.r_[np.full(6, 2.25), np.full(6, 100)]
    accepted = robot.apply_action(goal)
    np.testing.assert_allclose(accepted[:6], 0.025, rtol=1e-5)
    np.testing.assert_array_equal(robot._desired_joint_target_rad, goal[:6])
    assert robot._last_joint_target_deg.shape == (7,)
    robot.cr5.stop_motion.assert_not_called()


def test_home_goal_converges_over_multiple_cycles(robot, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("interface.time.monotonic", lambda: now[0])
    robot._last_joint_command_time = now[0] - 0.05
    robot.cr5.send_servoj.side_effect = lambda target: setattr(robot.cr5.joint_position, "return_value", target)
    goal = np.r_[np.full(6, 2.25), np.full(6, 100)]
    commands = []
    for _ in range(100):
        commands.append(robot.apply_action(goal)[:6])
        now[0] += 0.05
    np.testing.assert_allclose(commands[-1], goal[:6], atol=1e-6)
    assert np.max(np.abs(np.diff(commands, axis=0))) <= 0.025001


@pytest.mark.parametrize("direction", [-1, 1])
def test_reported_j4_feedback_lag_does_not_accumulate(robot, monkeypatch, direction):
    monkeypatch.setattr("interface.time.monotonic", lambda: 100.0)
    robot._last_joint_command_time = 99.95
    actual = np.array([0, 0, 0, 4.055, 0, 0, 17], dtype=np.float32)
    robot.cr5.joint_position.return_value = actual.tolist()
    previous = np.deg2rad(actual[:6])
    # Before the fix, advancing this accepted target by 0.025 rad reproduces
    # the reported 0.122652 rad candidate-to-feedback error on J4.
    previous[3] += direction * (0.12265210598707199 - 0.025)
    robot._last_joint_target_deg = np.r_[np.rad2deg(previous), 17]
    goal = np.r_[np.deg2rad(actual[:6]), [10, 20, 30, 40, 50, 60]]
    goal[3] += direction * np.deg2rad(18.363 - 4.055)

    accepted = robot.apply_action(goal)

    assert abs(accepted[3] - previous[3]) <= 0.025001
    assert abs(accepted[3] - np.deg2rad(actual[3])) < 0.097653
    np.testing.assert_array_equal(accepted[6:], goal[6:])
    np.testing.assert_array_equal(robot._desired_joint_target_rad, goal[:6].astype(np.float32))
    assert robot.cr5.send_servoj.call_args.args[0][6] == 17
    robot.cr5.stop_motion.assert_not_called()


def test_joint_tracker_waits_for_feedback_and_recovers(robot, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("interface.time.monotonic", lambda: now[0])
    robot._last_joint_command_time = now[0] - 0.05
    actual = np.zeros(6)
    previous = actual.copy()
    goal = np.r_[[0.4, -0.3, 0.2, -0.25, 0.15, -0.1], [10, 20, 30, 40, 50, 60]]
    for tick in range(450):
        if tick == 225:
            np.testing.assert_allclose(actual, goal[:6], atol=1e-5)
            goal[:6] *= -1
        robot.cr5.joint_position.return_value = np.r_[np.rad2deg(actual), 17].tolist()
        accepted = robot.apply_action(goal)
        assert np.max(np.abs(accepted[:6] - previous)) <= 0.025001
        assert np.max(np.abs(accepted[:6] - actual)) <= 0.025001
        np.testing.assert_array_equal(accepted[6:], goal[6:])
        # Feedback is initially stationary, then follows only 20% of the
        # remaining distance each cycle instead of instantly reaching targets.
        if tick >= 25:
            actual += 0.2 * (accepted[:6] - actual)
        previous = accepted[:6].copy()
        now[0] += 0.05
    np.testing.assert_allclose(actual, goal[:6], atol=1e-5)
    robot.cr5.stop_motion.assert_not_called()


def test_joint_tracker_rechecks_feedback_before_send(robot):
    robot.cr5.joint_position.side_effect = [
        [0, 0, 0, 0, 0, 0, 17],
        [0, 0, 0, -20, 0, 0, 17],
    ]
    with pytest.raises(RuntimeError, match="关节下发目标与反馈距离超限"):
        robot.apply_action(np.r_[np.full(6, 0.5), np.full(6, 100)])
    robot.cr5.send_servoj.assert_not_called()
    robot.cr5.stop_motion.assert_called_once()


def test_server_metadata_rejects_wrong_frequency():
    from main_rtc import _validate_server_metadata

    with pytest.raises(RuntimeError, match="control_hz"):
        _validate_server_metadata({"state_dim": 12, "physical_action_dim": 12, "control_hz": 30})


def test_websocket_inference_timeout_is_used(monkeypatch):
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    ws = Mock()
    ws.recv.return_value = msgpack_numpy.packb({"actions": np.zeros((20, 12))})
    monkeypatch.setattr(WebsocketClientPolicy, "_wait_for_server", lambda self: (ws, {}))
    client = WebsocketClientPolicy(inference_timeout_s=0.25)
    assert client.infer({})["actions"].shape == (20, 12)
    ws.recv.assert_called_once_with(timeout=0.25)


def test_actual_loader_config_contract():
    import dataclasses

    from openpi.training.data_loader import create_torch_dataset

    cfg = config.get_config("pi05_cr3_o6_joint_abs_lora")
    dc = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), repo_id="fake")
    dataset = create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    assert len(dataset) == 1024


def test_expired_goal_does_not_move(robot):
    with pytest.raises(TimeoutError):
        robot.apply_action(
            np.r_[np.full(6, 2.25), np.full(6, 100)], observation_timestamp_ns=time.monotonic_ns() - 1_000_000_000
        )
    robot.cr5.send_servoj.assert_not_called()
    robot.cr5.stop_motion.assert_called_once()


@pytest.mark.parametrize("feedback_delay_ms", [20, 60])
@pytest.mark.parametrize("control_hz", [20, 30])
def test_rtc_preserves_full_chunk_and_reports_deadline(robot, monkeypatch, caplog, feedback_delay_ms, control_hz):
    from interface import _ActionDeadlineExpired
    from main_rtc import AsyncDeltaRTCController
    from main_rtc import RTCDeploymentConfig
    from main_rtc import _InferenceResult

    now_ns = [100_300_000_000]
    monkeypatch.setattr("interface.time.monotonic_ns", lambda: now_ns[0])
    monkeypatch.setattr("interface.time.monotonic", lambda: now_ns[0] / 1e9)
    robot._last_joint_command_time = 100.25
    joint_reads = [0]

    def joint_feedback():
        joint_reads[0] += 1
        if joint_reads[0] == 2:
            # Expire the old row-0 deadline specifically inside servo_j(),
            # after its feedback RPC, as in the reported traceback.
            now_ns[0] += feedback_delay_ms * 1_000_000
        return [0, 0, 0, 0, 0, 0, 17]

    robot.cr5.joint_position.side_effect = joint_feedback
    actions = np.zeros((20, 12), dtype=np.float32)
    actions[:, :6] = np.arange(20)[:, None] / 1000
    actions[:, 6:] = np.arange(20)[:, None] + 100
    controller = AsyncDeltaRTCController(
        robot, Mock(), "offline deadline regression", RTCDeploymentConfig(control_hz=control_hz, request_lead_steps=6)
    )
    controller._pending_chunk = _InferenceResult(
        request_id=1, actions=actions, latency_s=0.25,
        observation_timestamp_ns=100_000_000_000,
        request_started_ns=100_000_000_000,
        observation_latency_s=0.04, ready_ns=100_290_000_000,
    )
    controller._promote_pending_if_boundary()
    action, scheduled_ns = controller._active_actions.popleft()
    if feedback_delay_ms == 20:
        accepted = robot.apply_action(action, observation_timestamp_ns=scheduled_ns)
        np.testing.assert_allclose(accepted, actions[0], rtol=1e-6)
        robot.cr5.send_servoj.assert_called_once()
        robot.o6.set_positions.assert_called_once()
        robot.cr5.stop_motion.assert_not_called()
    else:
        with pytest.raises(_ActionDeadlineExpired, match="ServoJ 下发前已过期") as error:
            robot.apply_action(action, observation_timestamp_ns=scheduled_ns)
        assert "feedback_rpc_ms=60.000" in str(error.value)
        assert "overdue_ms=10.000" in str(error.value)
        assert "entry_age_ms=300.000" in caplog.text
        assert "budget_before_servoj_ms=50.000" in caplog.text
        robot.cr5.send_servoj.assert_not_called()
        robot.o6.set_positions.assert_not_called()
        robot.cr5.stop_motion.assert_called_once()
    assert scheduled_ns == 100_000_000_000
    np.testing.assert_array_equal(action, actions[0])
    np.testing.assert_array_equal(np.array([row for row, _ in controller._active_actions]), actions[1:])
    assert [stamp for _, stamp in controller._active_actions] == [
        100_000_000_000 + round(index * 1e9 / control_hz) for index in range(1, 20)
    ]
