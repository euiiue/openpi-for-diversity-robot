import base64
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import socket
import sys
import time
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from camera_preview import CameraPreviewPolicy
from interface import warmup_policy
import main_rtc


@pytest.fixture
def observation():
    now = time.monotonic_ns()
    global_rgb = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    return {
        "observation/global_rgb": global_rgb,
        "observation/wrist_rgb": np.full((480, 640, 3), [240, 70, 10], dtype=np.uint8),
        "observation/right_wrist_rgb": np.full((480, 640, 3), [10, 70, 240], dtype=np.uint8),
        "observation/state": np.zeros(12, dtype=np.float32),
        "prompt": "离线测试 <script>not HTML</script>",
        "timestamp_ns": now - 50_000_000,
        "global_timestamp_ns": now - 30_000_000,
        "wrist_timestamp_ns": now - 10_000_000,
        "right_wrist_timestamp_ns": now - 20_000_000,
    }


@pytest.fixture
def preview():
    policy = Mock()
    policy.infer.return_value = {"actions": np.zeros((20, 12), dtype=np.float32)}
    preview = CameraPreviewPolicy(policy, 0)
    preview.start()
    yield preview
    preview.close()


def test_http_shows_224x224_model_previews_and_new_frames(preview, observation):
    connection = HTTPConnection(preview.url.removeprefix("http://"), timeout=3)
    try:
        connection.request("GET", "/snapshot")
        assert json.loads(connection.getresponse().read()) == {"sequence": 0, "preview_only": False}

        robot = Mock()
        robot.get_observation.return_value = observation
        warmup_policy(preview, robot, observation["prompt"])
        assert preview.policy.infer.call_args.args[0] is observation

        connection.request("GET", "/snapshot")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        snapshot = json.loads(response.read())
        assert snapshot["sequence"] == snapshot["completed"] == 1
        assert snapshot["preview_only"] is False
        assert snapshot["prompt"] == observation["prompt"]
        assert len(snapshot["images"]) == 3
        for view in snapshot["images"]:
            raw = base64.b64decode(view["src"].split(",", 1)[1])
            decoded = np.asarray(Image.open(io.BytesIO(raw)))
            source = observation[f"observation/{view['name']}"]
            assert decoded.shape == (224, 224, 3)
            assert view["shape"] == [224, 224, 3]
            assert view["source_shape"] == list(source.shape)
            assert view["capture_age_ms"] >= 0
            if view["name"] == "wrist_rgb":
                assert np.all(decoded == [240, 70, 10])
        assert snapshot["images"][0]["timestamp_ns"] == str(observation["global_timestamp_ns"])
        assert snapshot["images"][2]["timestamp_ns"] == str(observation["right_wrist_timestamp_ns"])

        second = dict(observation)
        second["observation/global_rgb"] = np.zeros((480, 640, 3), dtype=np.uint8)
        second["global_timestamp_ns"] += 1_000_000
        result = preview.infer(second)
        assert result is preview.policy.infer.return_value
        connection.request("GET", "/snapshot")
        next_snapshot = json.loads(connection.getresponse().read())
        assert next_snapshot["sequence"] == next_snapshot["completed"] == 2
        assert next_snapshot["images"][0]["src"] != snapshot["images"][0]["src"]
        assert next_snapshot["images"][0]["timestamp_ns"] == str(second["global_timestamp_ns"])

        connection.request("GET", "/")
        page = connection.getresponse().read().decode("utf-8")
        assert 'id="right_wrist_rgb"' in page
        assert 'id="roi_rgb"' not in page
        assert "observation/right_wrist_rgb → right_wrist_0_rgb" in page
        assert "模型输入 224×224 RGB" in page
        assert "旧帧" in page
    finally:
        connection.close()


def test_inference_does_not_encode_or_modify_input(preview, observation, monkeypatch):
    encode = Mock(side_effect=AssertionError("Encoding must only run in the HTTP thread"))
    monkeypatch.setattr(Image, "fromarray", encode)
    copies = {key: value.copy() for key, value in observation.items() if isinstance(value, np.ndarray)}
    result = preview.infer(observation)
    assert result is preview.policy.infer.return_value
    assert preview.policy.infer.call_args.args[0] is observation
    encode.assert_not_called()
    for key, value in copies.items():
        np.testing.assert_array_equal(observation[key], value)


def test_inference_failure_is_not_reported_as_completed(preview, observation):
    preview.policy.infer.side_effect = TimeoutError("inference timed out")
    with pytest.raises(TimeoutError, match="inference timed out"):
        preview.infer(observation)
    snapshot = preview.snapshot()
    assert snapshot["sequence"] == 1
    assert snapshot["completed"] == 0


def test_preview_only_status_is_available_before_first_inference():
    policy = Mock()
    preview = CameraPreviewPolicy(policy, 0, preview_only=True)
    preview.start()
    try:
        assert preview.snapshot() == {"sequence": 0, "preview_only": True}
        page = Path(__file__).resolve().parents[1] / "deploy" / "camera_preview.html"
        html = page.read_text(encoding="utf-8")
        assert "模型动作会被丢弃，不会发送机械臂运动命令" in html
        assert "模型动作会用于机器人控制" in html
    finally:
        preview.close()


def test_idle_browser_connection_does_not_block_page_or_frames(preview, observation):
    host, port = preview.url.removeprefix("http://").split(":")
    # Browsers may preconnect without sending a request. Leave that connection
    # open while a second connection fetches the page and the input images.
    with socket.create_connection((host, int(port)), timeout=2):
        connection = HTTPConnection(host, int(port), timeout=2)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200
            assert "三路模型输入" in response.read().decode("utf-8")
            preview.infer(observation)
            connection.request("GET", "/snapshot")
            snapshot = json.loads(connection.getresponse().read())
            assert snapshot["sequence"] == snapshot["completed"] == 1
            assert len(snapshot["images"]) == 3
        finally:
            connection.close()


@pytest.mark.parametrize("fail_control", [False, True])
def test_main_passes_preview_to_warmup_and_rtc_and_closes_it(monkeypatch, observation, fail_control):
    robot = Mock()
    robot.get_observation.return_value = observation
    websocket = Mock()
    websocket.get_server_metadata.return_value = {"control_hz": 20}
    preview = Mock()
    preview.infer.return_value = {"actions": np.zeros((20, 12), dtype=np.float32)}
    controller = Mock()
    if fail_control:
        controller.run.side_effect = TimeoutError("RTC expired")
    else:
        controller.run.return_value = {}
    create_controller = Mock(return_value=controller)
    monkeypatch.setattr(main_rtc, "NrcArmO6Interface", Mock(return_value=robot))
    monkeypatch.setattr(main_rtc, "CameraPreviewPolicy", Mock(return_value=preview))
    monkeypatch.setattr(main_rtc, "AsyncDeltaRTCController", create_controller)
    monkeypatch.setattr(main_rtc.websocket_client_policy, "WebsocketClientPolicy", Mock(return_value=websocket))
    monkeypatch.setattr(main_rtc, "validate_task_prompt", Mock())
    monkeypatch.setattr(main_rtc, "_validate_server_metadata", Mock())
    monkeypatch.setattr(sys, "argv", [
        "main_rtc.py", "--host", "127.0.0.1", "--config-dir",
        str(Path(__file__).resolve().parents[1] / "deploy" / "config"),
        "--prompt", observation["prompt"], "--preview-port", "8765", "--confirm-motion",
    ])
    if fail_control:
        with pytest.raises(TimeoutError, match="RTC expired"):
            main_rtc.main()
    else:
        main_rtc.main()
    robot.connect.assert_called_once_with(start_cameras=True)
    assert preview.infer.call_args.args[0] is observation
    assert create_controller.call_args.kwargs["policy"] is preview
    assert create_controller.call_args.kwargs["config"].control_hz == 20
    robot.enable_robot.assert_not_called()
    controller.stop.assert_called_once()
    robot.close.assert_called_once()
    preview.close.assert_called_once()


def test_preview_only_sends_observations_without_actuating_robot(monkeypatch, observation):
    robot = Mock()
    robot.get_observation.return_value = observation
    websocket = Mock()
    websocket.get_server_metadata.return_value = {"control_hz": 20}
    preview = Mock()
    preview.infer.return_value = {"actions": np.zeros((20, 12), dtype=np.float32)}
    controller_factory = Mock()
    monkeypatch.setattr(main_rtc, "NrcArmO6Interface", Mock(return_value=robot))
    monkeypatch.setattr(main_rtc, "CameraPreviewPolicy", Mock(return_value=preview))
    monkeypatch.setattr(main_rtc, "AsyncDeltaRTCController", controller_factory)
    monkeypatch.setattr(main_rtc.websocket_client_policy, "WebsocketClientPolicy", Mock(return_value=websocket))
    monkeypatch.setattr(main_rtc, "validate_task_prompt", Mock())
    monkeypatch.setattr(main_rtc, "_validate_server_metadata", Mock())
    monkeypatch.setattr(main_rtc.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    monkeypatch.setattr(sys, "argv", [
        "main_rtc.py", "--host", "127.0.0.1", "--config-dir",
        str(Path(__file__).resolve().parents[1] / "deploy" / "config"),
        "--prompt", observation["prompt"], "--preview-port", "8765", "--preview-only",
    ])

    main_rtc.main()

    robot.connect.assert_called_once_with(start_cameras=True)
    assert preview.infer.call_args.args[0] is observation
    controller_factory.assert_not_called()
    robot.enable_robot.assert_not_called()
    robot.start_control.assert_not_called()
    robot.apply_action.assert_not_called()
    robot.emergency_stop.assert_not_called()
    robot.close.assert_called_once()
    preview.close.assert_called_once()


@pytest.mark.parametrize("motion_flag", ["--confirm-motion", "--power-on"])
def test_preview_only_rejects_motion_flags_before_connecting(monkeypatch, observation, motion_flag):
    websocket_factory = Mock()
    monkeypatch.setattr(main_rtc.websocket_client_policy, "WebsocketClientPolicy", websocket_factory)
    monkeypatch.setattr(sys, "argv", [
        "main_rtc.py", "--host", "127.0.0.1", "--config-dir",
        str(Path(__file__).resolve().parents[1] / "deploy" / "config"),
        "--prompt", observation["prompt"], "--preview-port", "8765", "--preview-only",
        motion_flag,
    ])

    with pytest.raises(SystemExit, match="cannot be combined"):
        main_rtc.main()

    websocket_factory.assert_not_called()
