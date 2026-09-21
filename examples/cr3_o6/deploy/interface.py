"""Hardware interface for NRC-controlled CR3/CR5 + LinkerHand O6 + D435.

This module is the *robot-side* boundary of the OpenPI deployment.  It owns the
NRC sockets, the O6 Modbus bus, and (optionally) one or two RealSense cameras.
The policy process must not open any of those devices itself.

Physical contracts exposed by :class:`DobotCR5O6Interface`:

* state: 12 float32 values (CR3 q1..q6 + O6)
  ``[CR3 joint rad x6, O6 raw position x6]``
* action: 12 float32 values
  ``[CR3 joint absolute rad x6, O6 raw target x6]``

The NRC vendor binding in ``vendor/nrc_linux_x86_64`` is loaded lazily because
the supplied binary is tied to CPython 3.12.  This lets OpenPI tooling running
under Python 3.11 import this file for static inspection, while the real robot
client must run under Python 3.12.
"""

from __future__ import annotations

import ctypes
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
import struct
from dataclasses import dataclass
from dataclasses import fields
import importlib
import logging
import math
import os
from pathlib import Path
import platform
import sys
import sysconfig
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
NRC_VENDOR_DIR = THIS_DIR / "vendor" / "nrc_linux_x86_64"
NRC_PYTHON_FILE = NRC_VENDOR_DIR / "nrc_interface.py"
NRC_NATIVE_FILE = NRC_VENDOR_DIR / "_nrc_host.so"

NRC_RESULT_NAMES = {
    -6: "通信超时",
    -5: "SDK 异常",
    -4: "当前状态不允许该操作",
    -3: "参数错误",
    -2: "控制柜连接断开",
    -1: "接收控制柜响应失败",
    0: "成功",
}
NRC_SERVO_STATE_NAMES = {
    0: "停止",
    1: "就绪",
    2: "报警",
    3: "运行",
}


class _ActionDeadlineExpired(TimeoutError):
    """A model target expired before any new ServoJ command was sent."""

class _RosServoLBridge:
    def __init__(self):
        self.shm = shared_memory.SharedMemory(name="cr5_shared_state", create=False)
        resource_tracker.unregister(self.shm._name, "shared_memory")
        self.lock = threading.RLock()
        self.servoj_open = False

    def send(self, pose_mm, command_id=6):
        values = [float(v) for v in pose_mm] + [0.0]
        with self.lock:
            self.shm.buf[384 + 132] = 0
            struct.pack_into("i", self.shm.buf, 128, command_id)
            struct.pack_into("18d", self.shm.buf, 132, *(values + [0.0] * 11))
            self.shm.buf[128 + 148] = 1
            deadline = time.monotonic() + 1.0
            while self.shm.buf[384 + 132] != 1:
                if time.monotonic() >= deadline:
                    raise TimeoutError("ROS ServoL shared-memory response timeout")
                time.sleep(0.001)
            code = struct.unpack_from("i", self.shm.buf, 384)[0]
            message = bytes(self.shm.buf[388:516]).rstrip(b"\0").decode("utf-8")
            self.shm.buf[516] = 0
            if code != 0:
                raise RuntimeError(f"ROS ServoL command failed: {message}")

    def open_servoj(self, *args):
        self.send([0.0] * 6, command_id=4)
        self.servoj_open = True

    def power_on(self):
        self.send([0.0] * 6, command_id=1)

    def close_servoj(self):
        self.send([0.0] * 6, command_id=5)
        self.servoj_open = False

    def joint_position(self):
        return list(struct.unpack_from("6d", self.shm.buf, 1)) + [0.0]

    def tcp_position(self):
        return list(struct.unpack_from("6d", self.shm.buf, 49)) + [0.0]

    def connections_ready(self):
        return bool(self.shm.buf[0])

    def servo_state(self):
        return 3

    def close(self):
        self.shm.close()


def _finite_vector(values: Sequence[Any], size: int, name: str) -> np.ndarray:
    """Convert one public vector to a finite float32 array with fixed shape."""

    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,):
        raise ValueError(f"{name} 必须是 ({size},)，实际为 {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    return result


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _load_nrc_api() -> Any:
    """Load the vendored NRC binding without depending on TEST_PY12.

    The checked-in native library links against libpython3.12, so importing it
    in a different interpreter is unsafe even if Linux happens to load the .so.
    """

    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("当前 NRC 厂商库只支持 Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "NRC 实机客户端必须使用 Python 3.12；"
            f"当前为 Python {sys.version_info.major}.{sys.version_info.minor}"
        )
    for path in (NRC_PYTHON_FILE, NRC_NATIVE_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 NRC 厂商文件: {path}")

    # Some Python 3.12 distributions do not export libpython globally even
    # though the NRC extension declares it as a DT_NEEDED dependency.  Resolve
    # it from the *current interpreter* instead of hard-coding an old virtual
    # environment path or requiring LD_LIBRARY_PATH in every shell.
    py_major, py_minor = sys.version_info[:2]
    runtime_names = [
        f"libpython{py_major}.{py_minor}.so.1.0",
        f"libpython{py_major}.{py_minor}.so",
    ]
    configured_runtime = str(sysconfig.get_config_var("LDLIBRARY") or "")
    if ".so" in configured_runtime and configured_runtime not in runtime_names:
        runtime_names.append(configured_runtime)
    runtime_dirs = [
        NRC_VENDOR_DIR / "lib",
        Path(sys.executable).resolve().parent.parent / "lib",
        Path(sys.prefix).resolve() / "lib",
    ]
    configured_libdir = sysconfig.get_config_var("LIBDIR")
    if configured_libdir:
        runtime_dirs.append(Path(str(configured_libdir)))
    runtime_candidates = [
        directory / name for directory in runtime_dirs for name in runtime_names
    ]
    runtime_errors: list[str] = []
    for runtime_path in runtime_candidates:
        try:
            exists = runtime_path.is_file()
        except OSError as exc:
            runtime_errors.append(f"{runtime_path}: {exc}")
            continue
        if not exists:
            continue
        try:
            ctypes.CDLL(str(runtime_path), mode=ctypes.RTLD_GLOBAL)
            break
        except OSError as exc:
            runtime_errors.append(f"{runtime_path}: {exc}")

    existing = sys.modules.get("nrc_interface")
    if existing is not None:
        loaded_from = Path(str(getattr(existing, "__file__", ""))).resolve()
        if loaded_from.parent != NRC_VENDOR_DIR.resolve():
            raise RuntimeError(f"已加载其他位置的 nrc_interface: {loaded_from}")
        return existing

    vendor_text = str(NRC_VENDOR_DIR)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    try:
        module = importlib.import_module("nrc_interface")
    except ImportError as exc:
        runtime_detail = "; ".join(runtime_errors) or "当前 Python 环境未提供可加载的 libpython3.12"
        raise RuntimeError(
            "加载 NRC SDK 失败。请确认使用 Python 3.12，且当前 "
            f"Python 环境包含 libpython3.12.so.1.0。运行库检查: {runtime_detail}"
        ) from exc

    loaded_from = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded_from.parent != NRC_VENDOR_DIR.resolve():
        raise RuntimeError(f"NRC SDK 加载位置错误: {loaded_from}")
    return module


@dataclass(frozen=True)
class InterfaceConfig:
    """Machine-specific settings; no absolute project path belongs here."""

    controller_ip: str = "192.0.2.1"
    robot_num: int = 1
    command_port: int = 6001
    servo_port: int = 7000
    connection_timeout_s: float = 10.0

    servoj_vmax: float = 50.1
    servoj_amax: float = 100.0
    servoj_jmax: float = 200.0

    o6_port: str = "/dev/ttyUSB0"
    o6_baudrate: int = 115200
    o6_hand_id: int = 0x27
    o6_response_timeout_s: float = 0.15
    o6_frame_gap_s: float = 0.03
    o6_position_period_s: float = 0.20
    o6_fault_period_s: float = 0.80
    o6_feedback_max_age_s: float = 1.0
    o6_speed: tuple[int, ...] | None = None
    o6_torque: tuple[int, ...] | None = None

    # The three model image slots are independent RealSense camera streams.
    global_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    right_wrist_camera_serial: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_frame_max_age_s: float = 0.5
    camera_max_skew_s: float = 0.1

    # False allows orientation control under the configured action contract.  The optional lock remains
    # available for tasks that deliberately require a fixed tool orientation.
    lock_rpy: bool = False
    action_contract: str = "cr3_q1_q6_absolute_plus_o6_target_6d"

    # Deployment target tracker. The model's first six action dimensions are
    # collection-time tracking residuals, not per-cycle displacement commands:
    #     desired_tcp = current_feedback_tcp + action[:6]
    # These limits are therefore applied to successive commanded TCP targets,
    # not to the model residual itself. Defaults match the GELLO collection path.
    max_tcp_speed_mm_s: float = 150.0
    max_tcp_angular_speed_rad_s: float = 0.1
    tcp_tracker_max_dt_s: float = 0.05

    # Commanded TCP remains inside workspace_min/max. Actual feedback is allowed
    # a bounded tracking margin before emergency stop. This does NOT expand the
    # model/IK/ServoJ command workspace.
    feedback_workspace_margin_m: float = 0.030
    deployment_log_period_s: float = 0.50

    # Motion is intentionally disabled until real workspace bounds are supplied.
    workspace_min_m: tuple[float, float, float] | None = None
    workspace_max_m: tuple[float, float, float] | None = None
    # Site-reviewed limits are required before starting model-driven motion.
    max_joint_step_rad: float | None = None
    max_tracking_error_rad: float | None = None
    action_max_age_s: float = 0.35
    joint_target_max_speed_rad_s: float = 0.5

    @classmethod
    def from_yaml_dir(
        cls,
        config_dir: str | Path | None = None,
        **overrides: Any,
    ) -> "InterfaceConfig":
        """Load robot and camera settings from one portable directory.

        YAML keys intentionally use the same names as this dataclass. Unknown
        keys are rejected so a typo cannot silently disable a safety setting.
        Explicit keyword overrides are applied last for tests or CLI options.
        """

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "缺少 PyYAML；机器人 Python 3.12 环境需安装 pyyaml"
            ) from exc

        root = Path(config_dir).expanduser().resolve() if config_dir else THIS_DIR / "config"
        allowed = {field.name for field in fields(cls)}
        merged: dict[str, Any] = {}
        sources: dict[str, Path] = {}
        config_files = (
            ("robot.example.yaml", False),
            ("camera.example.yaml", False),
            ("robot.local.yaml", True),
            ("camera.local.yaml", True),
        )
        for filename, optional in config_files:
            path = root / filename
            if not path.is_file():
                if optional:
                    continue
                raise FileNotFoundError(f"缺少配置文件: {path}")
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"YAML 解析失败 {path}: {exc}") from exc
            if data is None:
                continue
            if not isinstance(data, dict):
                raise TypeError(f"{path} 顶层必须是 key: value 映射")
            unknown = sorted(set(data) - allowed)
            if unknown:
                raise ValueError(f"{path} 包含未知配置项: {unknown}")
            duplicate = sorted(set(data) & set(merged))
            if duplicate and not optional:
                previous = {key: str(sources[key]) for key in duplicate}
                raise ValueError(
                    f"{path} 重复定义 {duplicate}，之前来源={previous}"
                )
            merged.update(data)
            sources.update({key: path for key in data})

        unknown_overrides = sorted(set(overrides) - allowed)
        if unknown_overrides:
            raise ValueError(f"未知覆盖配置项: {unknown_overrides}")
        merged.update(overrides)

        tuple_keys = {
            "workspace_min_m",
            "workspace_max_m",
            "o6_speed",
            "o6_torque",
        }
        for key in tuple_keys & set(merged):
            if merged[key] is not None:
                merged[key] = tuple(merged[key])
        return cls(**merged)

    def __post_init__(self) -> None:
        if self.action_contract not in {
            "tcp_absolute_6d_plus_o6_target_6d", "cr3_q1_q6_absolute_plus_o6_target_6d"
        }:
            raise ValueError(f"Unsupported action contract: {self.action_contract}")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{field.name} 必须是有限数")
        if not isinstance(self.lock_rpy, bool):
            raise ValueError("lock_rpy 必须是布尔值")
        if (self.o6_speed is None) != (self.o6_torque is None):
            raise ValueError("o6_speed 和 o6_torque 必须同时设置")
        for name in ("o6_speed", "o6_torque"):
            values = getattr(self, name)
            if values is not None:
                vector = _finite_vector(values, 6, name)
                if np.any(vector < 0) or np.any(vector > 255) or np.any(vector != np.rint(vector)):
                    raise ValueError(f"{name} 必须包含六个 0..255 整数")
        if isinstance(self.robot_num, bool) or not isinstance(self.robot_num, int) or not 1 <= self.robot_num <= 4:
            raise ValueError("robot_num 必须为 1..4 的整数")
        for name in ("max_joint_step_rad", "max_tracking_error_rad"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} 必须是正有限数")
        for name in ("camera_max_skew_s", "action_max_age_s", "joint_target_max_speed_rad_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是正有限数")
        if not self.controller_ip.strip():
            raise ValueError("controller_ip 不能为空")
        if not (1 <= self.command_port <= 65535 and 1 <= self.servo_port <= 65535):
            raise ValueError("NRC 端口必须在 1..65535")
        if self.command_port == self.servo_port:
            raise ValueError("command_port 和 servo_port 不能相同")
        if self.connection_timeout_s <= 0:
            raise ValueError("connection_timeout_s 必须大于 0")
        if min(self.servoj_vmax, self.servoj_amax, self.servoj_jmax) <= 0:
            raise ValueError("ServoJ 速度/加速度/加加速度必须大于 0")
        if not self.o6_port.strip():
            raise ValueError("o6_port 不能为空")
        if not 1 <= self.o6_hand_id <= 247:
            raise ValueError("o6_hand_id 必须是有效 Modbus 地址")
        if min(
            self.o6_response_timeout_s,
            self.o6_frame_gap_s,
            self.o6_position_period_s,
            self.o6_fault_period_s,
            self.o6_feedback_max_age_s,
        ) <= 0:
            raise ValueError("O6 超时和周期必须大于 0")
        if min(self.camera_width, self.camera_height, self.camera_fps) <= 0:
            raise ValueError("相机宽、高、FPS 必须大于 0")
        if self.camera_frame_max_age_s <= 0:
            raise ValueError("camera_frame_max_age_s 必须大于 0")
        if self.max_tcp_speed_mm_s <= 0:
            raise ValueError("max_tcp_speed_mm_s 必须大于 0")
        if self.max_tcp_angular_speed_rad_s <= 0:
            raise ValueError("max_tcp_angular_speed_rad_s 必须大于 0")
        if self.tcp_tracker_max_dt_s <= 0:
            raise ValueError("tcp_tracker_max_dt_s 必须大于 0")
        if self.feedback_workspace_margin_m < 0:
            raise ValueError("feedback_workspace_margin_m 不能小于 0")
        if self.deployment_log_period_s <= 0:
            raise ValueError("deployment_log_period_s 必须大于 0")

        if (self.workspace_min_m is None) != (self.workspace_max_m is None):
            raise ValueError("workspace_min_m 和 workspace_max_m 必须同时设置")
        if self.workspace_min_m is not None and self.workspace_max_m is not None:
            lower = _finite_vector(self.workspace_min_m, 3, "workspace_min_m")
            upper = _finite_vector(self.workspace_max_m, 3, "workspace_max_m")
            if np.any(lower >= upper):
                raise ValueError("workspace_min_m 必须逐轴小于 workspace_max_m")

class _NrcAdapter:
    """Serialized access to the NRC 6001/7000 connection pair."""

    def __init__(self, api: Any, command_fd: int, servo_fd: int, robot_num: int = 1):
        self.api = api
        self.command_fd = int(command_fd)
        self.servo_fd = int(servo_fd)
        self.robot_num = robot_num
        self.lock = threading.RLock()
        self.servoj_open = False

    def _vector(self, values: Iterable[float]) -> Any:
        result = self.api.VectorDouble()
        for value in values:
            result.append(float(value))
        return result

    @staticmethod
    def _result_ok(result: Any) -> bool:
        try:
            return int(result) == 0
        except (TypeError, ValueError):
            return result == 0

    @staticmethod
    def _result_description(result: Any) -> str:
        try:
            code = int(result)
        except (TypeError, ValueError):
            return str(result)
        return f"{code}（{NRC_RESULT_NAMES.get(code, '未知返回码')}）"

    def _require_ok(self, result: Any, action: str) -> None:
        if not self._result_ok(result):
            raise RuntimeError(f"{action}失败: {self._result_description(result)}")

    def wait_connections_ready(self, timeout: float) -> None:
        for label, socket_fd in (
            ("6001 命令端口", self.command_fd),
            ("7000 跟踪端口", self.servo_fd),
        ):
            deadline = time.monotonic() + float(timeout)
            last_status: int | None = None
            while time.monotonic() < deadline:
                with self.lock:
                    last_status = int(self.api.get_connection_status(socket_fd))
                if last_status == 0:
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError(f"CR5 {label}连接未就绪，状态={last_status}")

    def connection_statuses(self) -> dict[str, int]:
        with self.lock:
            return {
                "6001": int(self.api.get_connection_status(self.command_fd)),
                "7000": int(self.api.get_connection_status(self.servo_fd)),
            }

    def connections_ready(self) -> bool:
        return all(value == 0 for value in self.connection_statuses().values())

    def _read_servo_state_unlocked(self) -> int:
        result = self.api.get_servo_state_robot(self.command_fd, self.robot_num, 0)
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError(f"CR5 伺服状态返回格式错误: {result}")
        self._require_ok(result[0], "读取 CR5 伺服状态")
        state = int(result[1])
        if state not in NRC_SERVO_STATE_NAMES:
            raise RuntimeError(f"CR5 返回未知伺服状态: {state}")
        return state

    def servo_state(self) -> int:
        with self.lock:
            return self._read_servo_state_unlocked()

    def _wait_servo_state(self, expected: set[int], timeout: float, action: str) -> int:
        deadline = time.monotonic() + float(timeout)
        last_state: int | None = None
        while time.monotonic() < deadline:
            last_state = self._read_servo_state_unlocked()
            if last_state in expected:
                return last_state
            time.sleep(0.05)
        expected_text = "/".join(
            f"{state}-{NRC_SERVO_STATE_NAMES.get(state, '未知')}" for state in sorted(expected)
        )
        detail = f"最后状态={last_state}"
        raise TimeoutError(f"{action}超时，期望 {expected_text}，{detail}")

    def power_on(self, timeout: float = 10.0) -> int:
        """Explicit state transition; never called automatically by connect()."""

        with self.lock:
            state = self._read_servo_state_unlocked()
            if state == 3:
                return state
            if state == 2:
                raise RuntimeError("NRC 处于报警状态；请先在控制柜排查，不自动清错或重新上电")
            if state == 0:
                self._require_ok(
                    self.api.set_servo_state_robot(self.command_fd, self.robot_num, 1),
                    "CR5 设置伺服就绪",
                )
                state = self._wait_servo_state({1}, timeout, "CR5 进入就绪状态")
            if state != 1:
                raise RuntimeError(
                    f"CR5 无法上使能：状态={state}-"
                    f"{NRC_SERVO_STATE_NAMES.get(state, '未知')}"
                )
            mode = self.api.get_current_mode_robot(self.command_fd, self.robot_num, 0)
            self._require_ok(mode[0], "读取 NRC 运行模式")
            if int(mode[1]) != 2:
                raise RuntimeError("请先在控制柜切换到运行模式 2，再人工上使能")
            self._require_ok(
                self.api.set_servo_poweron_robot(self.command_fd, self.robot_num), "NRC 伺服上电"
            )
            return self._wait_servo_state({3}, timeout, "CR5 进入运行状态")

    def running_state(self) -> int:
        with self.lock:
            result = self.api.get_robot_running_state_robot(self.command_fd, self.robot_num, 0)
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError(f"读取 CR5 运行状态返回格式错误: {result}")
        self._require_ok(result[0], "读取 CR5 运行状态")
        return int(result[1])

    def position(self, coord: int) -> list[float]:
        output = self.api.VectorDouble()
        with self.lock:
            result = self.api.get_current_position_robot(self.command_fd, self.robot_num, int(coord), output)
        values = [float(value) for value in output]
        self._require_ok(result, f"读取 CR5 坐标(coord={coord})")
        if len(values) < 7 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"CR5 坐标反馈不足 7 维: {values}")
        return values[:7]

    def joint_position(self) -> list[float]:
        return self.position(0)

    def tcp_position(self) -> list[float]:
        return self.position(1)

    def open_servoj(self, vmax: float, amax: float, jmax: float) -> None:
        if self.servoj_open:
            return
        with self.lock:
            result = self.api.open_servoJ(
                self.servo_fd,
                self._vector([vmax] * 7),
                self._vector([amax] * 7),
                self._vector([jmax] * 7),
            )
        self._require_ok(result, "开启 CR5 ServoJ")
        self.servoj_open = True

    def stop_servoj(self) -> None:
        if not self.servoj_open:
            return
        with self.lock:
            result = self.api.stop_servoJ(self.servo_fd)
        self._require_ok(result, "停止 CR5 ServoJ")
        self.servoj_open = False

    def inverse_kinematics(self, tcp_nrc: Sequence[float]) -> list[float]:
        if len(tcp_nrc) != 7:
            raise ValueError("NRC TCP 目标必须是 7 维")
        target = self._vector(tcp_nrc)
        joints = self.api.VectorDouble(7)
        with self.lock:
            result = self.api.get_origin_coord_to_target_coord_robot(
                self.command_fd, self.robot_num, 1, target, 0, joints
            )
        values = [float(value) for value in joints]
        self._require_ok(result, "CR5 逆运动学")
        if len(values) < 7:
            raise RuntimeError(f"CR5 逆解结果不足 7 维: {values}")
        return values[:7]

    def forward_kinematics(self, joints_deg: Sequence[float]) -> np.ndarray:
        """NRC coordinate 0 (joint degrees) to 1 (Cartesian mm/rad)."""
        output = self.api.VectorDouble(7)
        with self.lock:
            result = self.api.get_origin_coord_to_target_coord_robot(
                self.command_fd, self.robot_num, 0, self._vector(joints_deg), 1, output
            )
        self._require_ok(result, "CR3 forward kinematics")
        return _finite_vector(list(output), 7, "NRC forward kinematics")

    def send_servoj(self, joints_deg: Sequence[float]) -> None:
        if not self.servoj_open:
            raise RuntimeError("CR5 ServoJ 尚未开启")
        if len(joints_deg) != 7:
            raise ValueError("CR5 ServoJ 目标必须是 7 维")
        with self.lock:
            result = self.api.set_servoJ_pos(self.servo_fd, self._vector(joints_deg))
        self._require_ok(result, "发送 CR5 ServoJ")

    def stop_motion(self) -> None:
        errors = []
        try:
            self.stop_servoj()
        except Exception as exc:
            errors.append(exc)
        try:
            with self.lock:
                result = self.api.queue_motion_stop_not_power_off_robot(self.command_fd, self.robot_num)
            self._require_ok(result, "停止 NRC 运动队列")
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise ExceptionGroup("NRC 停止未完成", errors)


class _O6Worker:
    """Single-owner Modbus worker; the 20 Hz control loop never touches serial."""

    def __init__(self, config: InterfaceConfig):
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.RLock()
        self._position: tuple[int, ...] | None = None
        self._fault: tuple[int, ...] | None = None
        self._timestamp = 0.0
        self._desired_target: tuple[int, ...] | None = None
        self._last_sent_target: tuple[int, ...] | None = None
        self._error = ""
        self._phase = "未启动"
        self._profile_pending = False
        self._profile_ready = threading.Event()
        self._close_error: Exception | None = None

    @property
    def connected(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive() and not self.error)

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @property
    def last_sent_target(self) -> tuple[int, ...] | None:
        with self._lock:
            return self._last_sent_target

    def latest(self) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None, float]:
        with self._lock:
            return self._position, self._fault, self._timestamp

    def connect(self, timeout: float = 10.0) -> tuple[int, ...]:
        if self.connected:
            position, _, _ = self.latest()
            if position is None:
                raise RuntimeError("O6 已连接但没有位置反馈")
            return position
        if self._thread is not None:
            self.close()
        port = Path(self.config.o6_port)
        if not port.exists():
            raise FileNotFoundError(f"O6 串口不存在: {port}")
        if not os.access(port, os.R_OK | os.W_OK):
            raise PermissionError(f"O6 串口无读写权限: {port}")

        self._stop.clear()
        self._ready.clear()
        self._close_error = None
        with self._lock:
            self._position = None
            self._fault = None
            self._timestamp = 0.0
            self._desired_target = None
            self._last_sent_target = None
            self._profile_pending = False
            self._profile_ready.clear()
            self._error = ""
            self._phase = "启动通信线程"
        self._thread = threading.Thread(target=self._run, name="O6-Modbus", daemon=True)
        self._thread.start()
        if not self._ready.wait(float(timeout)):
            phase = self.phase
            self.close()
            raise TimeoutError(f"O6 初始化超时（阶段={phase}）")
        if self.error:
            error = self.error
            self.close()
            raise RuntimeError(error)
        position, _, _ = self.latest()
        if position is None:
            raise RuntimeError("O6 初始化后没有位置反馈")
        return position

    def get_positions(self, max_age_s: float) -> np.ndarray:
        if self.error:
            raise RuntimeError(self.error)
        position, fault, timestamp = self.latest()
        if position is None or len(position) != 6:
            raise RuntimeError("O6 位置反馈不完整")
        if fault is None or len(fault) != 6:
            raise RuntimeError("O6 故障反馈不完整")
        if any(int(value) != 0 for value in fault):
            raise RuntimeError(f"O6 电机故障: {fault}")
        age = time.monotonic() - timestamp
        if age > float(max_age_s):
            raise TimeoutError(f"O6 位置反馈已过期 {age:.3f}s")
        return np.asarray(position, dtype=np.float32)

    def set_positions(self, target: Sequence[Any]) -> tuple[int, ...]:
        converted = _finite_vector(target, 6, "O6 目标")
        command = tuple(int(value) for value in np.clip(np.rint(converted), 0, 255))
        if not self.connected:
            raise RuntimeError(self.error or "O6 尚未连接")
        with self._lock:
            self._desired_target = command
        return command

    def configure_motion(self) -> None:
        """Set the collector's explicit profile only on operator motion start."""
        if self.config.o6_speed is None:
            return
        if not self.connected:
            raise RuntimeError(self.error or "O6 尚未连接")
        with self._lock:
            self._profile_ready.clear()
            self._profile_pending = True
        if not self._profile_ready.wait(self.config.connection_timeout_s):
            raise TimeoutError("O6 速度/力矩设置超时")
        if self.error:
            raise RuntimeError(self.error)

    def hold_current(self) -> None:
        position, _, _ = self.latest()
        if position is not None and self.connected:
            with self._lock:
                self._desired_target = position

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            with self._lock:
                self._error = f"O6 通信线程无法退出（阶段={self._phase}）"
            raise RuntimeError(self.error)
        self._thread = None
        if self._close_error is not None:
            raise RuntimeError("O6 串口关闭失败") from self._close_error
        with self._lock:
            if not self._error:
                self._phase = "已关闭"

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._phase = "异常"
        self._ready.set()
        self._profile_ready.set()

    def _run(self) -> None:
        client: Any = None
        last_frame_time = 0.0

        def wait_bus() -> None:
            nonlocal last_frame_time
            remaining = self.config.o6_frame_gap_s - (time.perf_counter() - last_frame_time)
            if remaining > 0:
                self._stop.wait(remaining)

        def read_registers(address: int, count: int) -> tuple[int, ...]:
            nonlocal last_frame_time
            wait_bus()
            response = client.read_input_registers(
                address=address,
                count=count,
                slave=self.config.o6_hand_id,
            )
            last_frame_time = time.perf_counter()
            if response is None or response.isError():
                raise RuntimeError(
                    f"Modbus Read Failed (Addr={address}, Count={count}): {response}"
                )
            registers = getattr(response, "registers", None)
            if registers is None or len(registers) != count:
                raise RuntimeError(f"O6 Modbus 返回长度错误: {registers}")
            return tuple(int(value) for value in registers)

        def write_registers(address: int, values: tuple[int, ...]) -> None:
            nonlocal last_frame_time
            wait_bus()
            response = client.write_registers(
                address=address,
                values=[int(value) for value in values],
                slave=self.config.o6_hand_id,
            )
            last_frame_time = time.perf_counter()
            if response is None or response.isError():
                raise RuntimeError(
                    f"Modbus Write Failed (Addr={address}, Values={values}): {response}"
                )

        try:
            try:
                from pymodbus.client import ModbusSerialClient
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 pymodbus；机器人 Python 3.12 环境需安装 pymodbus==3.5.1"
                ) from exc

            with self._lock:
                self._phase = "打开 RS485 串口"
            client = ModbusSerialClient(
                port=self.config.o6_port,
                baudrate=self.config.o6_baudrate,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=self.config.o6_response_timeout_s,
                handle_local_echo=False,
            )
            if not client.connect():
                raise ConnectionError(f"RS485 connect fail to {self.config.o6_port}")

            with self._lock:
                self._phase = "读取 O6 初始状态"
            position = read_registers(0, 6)  # FC04: position
            fault = read_registers(24, 6)  # FC04: fault codes
            with self._lock:
                self._position = position
                self._fault = fault
                self._timestamp = time.monotonic()
                self._desired_target = position
                self._last_sent_target = position
                self._phase = "运行中"
            self._ready.set()

            next_position = time.monotonic() + self.config.o6_position_period_s
            next_fault = time.monotonic() + self.config.o6_fault_period_s
            while not self._stop.is_set():
                with self._lock:
                    profile_pending = self._profile_pending
                    target = self._desired_target
                    last_target = self._last_sent_target
                if profile_pending:
                    write_registers(6, self.config.o6_torque)
                    write_registers(12, self.config.o6_speed)
                    with self._lock:
                        self._profile_pending = False
                    self._profile_ready.set()
                if target is not None and target != last_target:
                    write_registers(0, target)  # FC16: six position targets
                    with self._lock:
                        self._last_sent_target = target
                    # Log only meaningful moves (>=3 register counts) so the log
                    # shows exactly when the gripper is told to open/close:
                    # demonstrated open ~[157, 95, 175], grasp ~[78, 85, 123].
                    if last_target is None or max(
                        abs(int(new) - int(old)) for new, old in zip(target, last_target)
                    ) >= 3:
                        logger.info(
                            "O6 hand target -> %s (was %s)",
                            list(target),
                            None if last_target is None else list(last_target),
                        )

                now = time.monotonic()
                if now >= next_position:
                    position = read_registers(0, 6)
                    with self._lock:
                        self._position = position
                        self._timestamp = time.monotonic()
                    next_position = time.monotonic() + self.config.o6_position_period_s
                if now >= next_fault:
                    fault = read_registers(24, 6)
                    with self._lock:
                        self._fault = fault
                    next_fault = time.monotonic() + self.config.o6_fault_period_s
                self._stop.wait(0.005)
        except Exception as exc:
            self._set_error(f"O6 通信线程异常: {type(exc).__name__}: {exc}")
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    self._close_error = exc
                    self._set_error(f"{self.error}; O6 串口关闭失败: {exc}")
            self._ready.set()


class _LatestRealSenseCamera:
    """One RealSense pipeline with a latest-frame buffer and no control-loop blocking."""

    def __init__(
        self,
        name: str,
        serial: str,
        width: int,
        height: int,
        fps: int,
    ):
        self.name = name
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.RLock()
        self._image: np.ndarray | None = None
        self._timestamp_ns = 0
        self._error = ""
        self._close_error: Exception | None = None

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def connected(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive() and not self.error)

    def start(self, timeout: float = 8.0) -> None:
        if self.connected:
            return
        if self._thread is not None:
            self.close()
        self._stop.clear()
        self._close_error = None
        self._ready.clear()
        with self._lock:
            self._image = None
            self._timestamp_ns = 0
            self._error = ""
        self._thread = threading.Thread(
            target=self._run,
            name=f"D435-{self.name}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(float(timeout)):
            self.close()
            raise TimeoutError(f"{self.name} D435 首帧超时")
        if self.error:
            error = self.error
            self.close()
            raise RuntimeError(error)

    def latest(self, max_age_s: float) -> tuple[np.ndarray, int]:
        with self._lock:
            image = self._image
            timestamp_ns = self._timestamp_ns
            error = self._error
        if error:
            raise RuntimeError(error)
        if image is None or timestamp_ns <= 0:
            raise RuntimeError(f"{self.name} D435 尚无图像")
        age_s = (time.monotonic_ns() - timestamp_ns) / 1_000_000_000.0
        if age_s > float(max_age_s):
            raise TimeoutError(f"{self.name} D435 图像已过期 {age_s:.3f}s")
        return image.copy(), timestamp_ns

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError(f"{self.name} D435 线程未退出")
        self._thread = None
        if self._close_error is not None:
            raise RuntimeError(f"{self.name} D435 pipeline 关闭失败") from self._close_error

    def _run(self) -> None:
        pipeline: Any = None
        try:
            try:
                import pyrealsense2 as rs
            except ImportError as exc:
                raise RuntimeError("机器人环境未安装 pyrealsense2") from exc

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )
            pipeline.start(config)

            consecutive_timeouts = 0
            while not self._stop.is_set():
                try:
                    frames = pipeline.wait_for_frames(1000)
                except RuntimeError as exc:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 5:
                        raise RuntimeError("连续 5 次等待图像超时") from exc
                    continue
                consecutive_timeouts = 0
                color = frames.get_color_frame()
                if not color:
                    continue
                bgr = np.asanyarray(color.get_data())
                if bgr.shape != (self.height, self.width, 3):
                    raise RuntimeError(f"D435 图像尺寸异常: {bgr.shape}")
                rgb = np.ascontiguousarray(bgr[..., ::-1])
                with self._lock:
                    self._image = rgb
                    self._timestamp_ns = time.monotonic_ns()
                self._ready.set()
        except Exception as exc:
            with self._lock:
                self._error = f"{self.name} D435 异常: {type(exc).__name__}: {exc}"
            self._ready.set()
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception as exc:
                    self._close_error = exc
                    with self._lock:
                        self._error = f"{self._error}; D435 pipeline 关闭失败: {exc}"


def validate_task_prompt(metadata: dict[str, Any], prompt: str) -> None:
    """Reject a deployment task that differs from the served training config."""
    expected = metadata.get("task_prompt")
    if not prompt.strip() or (expected is not None and expected != prompt):
        raise ValueError(f"Policy task prompt mismatch: server={expected!r}, client={prompt!r}")


def warmup_policy(policy: Any, robot: Any, prompt: str) -> None:
    """Compile inference on a read-only observation before enabling control.

    The warmup result is deliberately discarded; control fetches a new
    observation after initialization instead of executing an expired action.
    """
    result = policy.infer(robot.get_observation(prompt))
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 12 or not np.isfinite(actions).all():
        raise ValueError(f"Invalid policy warmup action chunk: {actions.shape}")
    logger.info("Policy warmup complete; discarded initialization actions")


class DobotCR5O6Interface:
    """The only hardware interface imported by collection/deployment code."""

    STATE_DIM = 12
    ACTION_DIM = 12
    ACTION_CONTRACT = "cr3_q1_q6_absolute_plus_o6_target_6d"

    def __init__(self, config: InterfaceConfig | None = None):
        self.config = config or InterfaceConfig.from_yaml_dir()
        self.ACTION_CONTRACT = self.config.action_contract
        self.command_fd = -1
        self.servo_fd = -1
        self._nrc_api: Any = None
        self.cr5: _NrcAdapter | None = None
        self.o6 = _O6Worker(self.config)
        self._global_camera = (
            _LatestRealSenseCamera(
                "global",
                self.config.global_camera_serial,
                self.config.camera_width,
                self.config.camera_height,
                self.config.camera_fps,
            )
            if self.config.global_camera_serial
            else None
        )
        self._wrist_camera = (
            _LatestRealSenseCamera(
                "wrist",
                self.config.wrist_camera_serial,
                self.config.camera_width,
                self.config.camera_height,
                self.config.camera_fps,
            )
            if self.config.wrist_camera_serial
            else None
        )
        self._right_wrist_camera = (
            _LatestRealSenseCamera(
                "right_wrist",
                self.config.right_wrist_camera_serial,
                self.config.camera_width,
                self.config.camera_height,
                self.config.camera_fps,
            )
            if self.config.right_wrist_camera_serial
            else None
        )
        self._control_started = False
        self._locked_rpy_rad: np.ndarray | None = None
        self._last_error = ""
        self._lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._last_joint_target_deg: np.ndarray | None = None
        self._desired_joint_target_rad: np.ndarray | None = None
        self._last_joint_command_time: float | None = None
        self._target_observation_timestamp_ns: int | None = None
        self._desired_tcp: np.ndarray | None = None
        self._commanded_tcp: np.ndarray | None = None
        self._desired_hand_target: np.ndarray | None = None
        self._last_tcp_command_time: float | None = None
        self._last_raw_desired_tcp: np.ndarray | None = None
        self._last_motion_log_time: float = 0.0
        self._last_joint_log_time: float = 0.0
        self._ros_servol: _RosServoLBridge | None = None

    def __enter__(self) -> "DobotCR5O6Interface":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _require_cr5(self) -> _NrcAdapter:
        robot = self.cr5
        if robot is None:
            raise RuntimeError("CR5 尚未连接")
        return robot

    def connect(self, *, start_cameras: bool = True) -> dict[str, Any]:
        """Connect all configured devices, but never enable or move the CR5."""

        with self._lock:
            if self.cr5 is not None:
                return self.get_diagnostics()
            self._last_error = ""

        api = _load_nrc_api()
        command_fd = -1
        servo_fd = -1
        try:
            if self.config.action_contract == "tcp_absolute_6d_plus_o6_target_6d":
                self._ros_servol = _RosServoLBridge()
                self.cr5 = self._ros_servol
                self.o6.connect(timeout=self.config.connection_timeout_s)
                if start_cameras:
                    if self._global_camera is not None: self._global_camera.start()
                    if self._wrist_camera is not None: self._wrist_camera.start()
                    if self._right_wrist_camera is not None: self._right_wrist_camera.start()
                self.get_state()
                return self.get_diagnostics()
            command_fd = int(
                api.connect_robot(self.config.controller_ip, str(self.config.command_port))
            )
            servo_fd = int(
                api.connect_robot(self.config.controller_ip, str(self.config.servo_port))
            )
            if command_fd <= 0 or servo_fd <= 0:
                raise ConnectionError(f"NRC 连接失败: 6001={command_fd}, 7000={servo_fd}")

            robot = _NrcAdapter(api, command_fd, servo_fd, self.config.robot_num)
            robot.wait_connections_ready(self.config.connection_timeout_s)
            self.o6.connect(timeout=self.config.connection_timeout_s)
            if start_cameras:
                if self._global_camera is not None:
                    self._global_camera.start()
                if self._wrist_camera is not None:
                    self._wrist_camera.start()
                if self._right_wrist_camera is not None:
                    self._right_wrist_camera.start()

            with self._lock:
                self._nrc_api = api
                self.command_fd = command_fd
                self.servo_fd = servo_fd
                self.cr5 = robot
            # Validate the complete physical state before reporting success.
            self.get_state()
            return self.get_diagnostics()
        except Exception as exc:
            self._last_error = f"连接设备失败: {type(exc).__name__}: {exc}"
            # Preserve handles locally as connect() may have failed before assignment.
            self._nrc_api = api
            self.command_fd = command_fd
            self.servo_fd = servo_fd
            try:
                self.close()
            except Exception as cleanup_error:
                raise ExceptionGroup("连接失败且清理未完成", [exc, cleanup_error]) from exc
            raise RuntimeError(self._last_error) from exc

    def enable_robot(self, timeout: float = 10.0) -> int:
        """Explicitly power on the CR5; connect() deliberately does not call this."""

        return self._require_cr5().power_on(timeout=timeout)

    def start_control(self) -> None:
        """Initialize the target tracker and open ServoJ once."""

        if self._control_started:
            raise RuntimeError("控制已启动，请先停止")
        if self.config.max_joint_step_rad is None or self.config.max_tracking_error_rad is None:
            raise RuntimeError("请先配置现场确认的 max_joint_step_rad / max_tracking_error_rad")
        if self.config.workspace_min_m is None or self.config.workspace_max_m is None:
            raise RuntimeError(
                "尚未配置真实 workspace_min_m/workspace_max_m，禁止启动运动"
            )
        robot = self._require_cr5()
        if not robot.connections_ready():
            raise RuntimeError("ROS CR5 状态桥未就绪：请确认 nrc_driver_node 与 cr5_ros_node 使用相同 ROS_LOCALHOST_ONLY")
        if robot.servo_state() != 3:
            raise RuntimeError("CR5 尚未处于伺服运行状态 3")

        current_hand = self.get_o6_positions()
        current_tcp = self.get_tcp_pose()
        lower = np.asarray(self.config.workspace_min_m, dtype=np.float32)
        upper = np.asarray(self.config.workspace_max_m, dtype=np.float32)
        if np.any(current_tcp[:3] < lower) or np.any(current_tcp[:3] > upper):
            below_mm = (current_tcp[:3] - lower) * 1000.0
            above_mm = (upper - current_tcp[:3]) * 1000.0
            raise RuntimeError(
                "当前 TCP 位于工作空间之外: "
                f"current_mm={np.round(current_tcp[:3] * 1000.0, 3).tolist()}, "
                f"min_mm={np.round(lower * 1000.0, 3).tolist()}, "
                f"max_mm={np.round(upper * 1000.0, 3).tolist()}, "
                f"margin_from_min_mm={np.round(below_mm, 3).tolist()}, "
                f"margin_to_max_mm={np.round(above_mm, 3).tolist()}"
            )
        current_joint_deg = _finite_vector(robot.joint_position(), 7, "NRC 关节反馈")
        self._last_joint_target_deg = current_joint_deg.copy()
        self._last_joint_command_time = time.monotonic()
        self._target_observation_timestamp_ns = None
        self._desired_tcp = current_tcp.copy()
        self._commanded_tcp = current_tcp.copy()
        self._desired_hand_target = current_hand.copy()
        self._last_tcp_command_time = time.monotonic()
        margin_m = float(self.config.feedback_workspace_margin_m)
        logger.info(
            "DEPLOY workspace command_min_mm=%s command_max_mm=%s "
            "feedback_emergency_min_mm=%s feedback_emergency_max_mm=%s",
            np.round(lower * 1000.0, 3).tolist(),
            np.round(upper * 1000.0, 3).tolist(),
            np.round((lower - margin_m) * 1000.0, 3).tolist(),
            np.round((upper + margin_m) * 1000.0, 3).tolist(),
        )

        self.o6.configure_motion()
        if self.config.lock_rpy:
            self._locked_rpy_rad = current_tcp[3:6].copy()
        else:
            self._locked_rpy_rad = None

        if self.config.action_contract == "tcp_absolute_6d_plus_o6_target_6d" and hasattr(robot, "power_on"):
            robot.power_on()
        robot.open_servoj(
            self.config.servoj_vmax,
            self.config.servoj_amax,
            self.config.servoj_jmax,
        )
        self._control_started = True

    def stop_control(self) -> None:
        self.emergency_stop()

    def is_ready(self) -> bool:
        try:
            robot = self._require_cr5()
            self.get_o6_positions()
            return robot.connections_ready() and robot.servo_state() == 3
        except Exception as exc:
            self._last_error = str(exc)
            return False

    def get_joint_positions(self) -> np.ndarray:
        """Return the six physical CR5 joints in radians."""

        raw_deg = np.asarray(self._require_cr5().joint_position()[:6], dtype=np.float32)
        if raw_deg.shape != (6,) or not np.all(np.isfinite(raw_deg)):
            raise RuntimeError(f"CR5 关节反馈无效: {raw_deg}")
        return np.deg2rad(raw_deg).astype(np.float32)

    def get_tcp_pose(self) -> np.ndarray:
        """Return [x,y,z,roll,pitch,yaw] in metres and radians."""

        raw = np.asarray(self._require_cr5().tcp_position()[:6], dtype=np.float32)
        if raw.shape != (6,) or not np.all(np.isfinite(raw)):
            raise RuntimeError(f"CR5 TCP 反馈无效: {raw}")
        # This NRC SDK returns TCP XYZ in mm and ABC in radians. Joint
        # coordinates alone use degrees, as in the GELLO collection runtime.
        return np.concatenate([raw[:3] / 1000.0, raw[3:6]]).astype(np.float32)

    def get_o6_positions(self) -> np.ndarray:
        """Return the six O6 positions in the native 0..255 order."""

        return self.o6.get_positions(self.config.o6_feedback_max_age_s)

    def get_state(self) -> np.ndarray:
        """Return the 12D joint state used by training and inference."""

        state = np.concatenate(
            [self.get_joint_positions(), self.get_o6_positions()]
        ).astype(np.float32)
        if state.shape != (self.STATE_DIM,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"12D state 无效: shape={state.shape}, value={state}")
        return state

    def get_images(self) -> dict[str, Any]:
        cameras = (self._global_camera, self._wrist_camera, self._right_wrist_camera)
        if any(camera is None for camera in cameras):
            raise RuntimeError("必须配置 global、wrist、right_wrist 三路相机")
        global_rgb, global_timestamp_ns = self._global_camera.latest(
            self.config.camera_frame_max_age_s
        )
        wrist_rgb, wrist_timestamp_ns = self._wrist_camera.latest(
            self.config.camera_frame_max_age_s
        )
        right_wrist_rgb, right_wrist_timestamp_ns = self._right_wrist_camera.latest(
            self.config.camera_frame_max_age_s
        )
        timestamps = (global_timestamp_ns, wrist_timestamp_ns, right_wrist_timestamp_ns)
        skew_s = (max(timestamps) - min(timestamps)) / 1e9
        if skew_s > self.config.camera_max_skew_s:
            raise TimeoutError(f"三路相机时间差 {skew_s:.3f}s 超过 {self.config.camera_max_skew_s}s")
        return {
            "global_rgb": global_rgb,
            "wrist_rgb": wrist_rgb,
            "right_wrist_rgb": right_wrist_rgb,
            "global_timestamp_ns": global_timestamp_ns,
            "wrist_timestamp_ns": wrist_timestamp_ns,
            "right_wrist_timestamp_ns": right_wrist_timestamp_ns,
        }

    def get_observation(self, prompt: str) -> dict[str, Any]:
        """Build one unnormalized OpenPI client observation."""

        if not str(prompt).strip():
            raise ValueError("prompt 不能为空")
        observation_started_ns = time.monotonic_ns()
        state = self.get_state()
        images = self.get_images()
        observation: dict[str, Any] = {
            "observation/global_rgb": images["global_rgb"],
            "observation/right_wrist_rgb": images["right_wrist_rgb"],
            "observation/state": state,
            "prompt": str(prompt),
            "timestamp_ns": observation_started_ns,
            "global_timestamp_ns": images["global_timestamp_ns"],
            "wrist_timestamp_ns": images["wrist_timestamp_ns"],
            "right_wrist_timestamp_ns": images["right_wrist_timestamp_ns"],
        }
        observation["observation/wrist_rgb"] = images["wrist_rgb"]
        observation["timestamp_ns"] = min(
            observation_started_ns,
            images["global_timestamp_ns"],
            images["wrist_timestamp_ns"],
            images["right_wrist_timestamp_ns"],
        )
        return observation

    def _filter_action(
        self,
        action: Sequence[Any],
        current_tcp: np.ndarray,
        current_hand: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct the collection-time commanded TCP from one model residual.

        Collection semantics are:
            action[:3]   = target_xyz - feedback_xyz
            action[3:6]  = shortest-angle(target_rpy - feedback_rpy)
            action[6:12] = absolute O6 target

        The model residual itself is intentionally *not* clipped. Safety rate
        limiting is applied later to target-to-target progression.
        """

        command = _finite_vector(action, self.ACTION_DIM, "action").copy()
        delta_tcp = command[:6].copy()
        hand_target = command[6:12].copy()

        lower = np.asarray(self.config.workspace_min_m, dtype=np.float32)
        upper = np.asarray(self.config.workspace_max_m, dtype=np.float32)

        # The strict command workspace is the training-data envelope. Actual TCP
        # feedback may temporarily differ from the commanded target because of
        # tracking lag. Only stop when feedback exits a separate emergency
        # envelope; never widen the desired/commanded target workspace itself.
        feedback_margin_m = float(self.config.feedback_workspace_margin_m)
        feedback_lower = lower - feedback_margin_m
        feedback_upper = upper + feedback_margin_m
        if (
            np.any(current_tcp[:3] < feedback_lower)
            or np.any(current_tcp[:3] > feedback_upper)
        ):
            raise RuntimeError(
                "当前 TCP 超出 feedback emergency workspace: "
                f"current_mm={np.round(current_tcp[:3] * 1000.0, 3).tolist()}, "
                f"command_min_mm={np.round(lower * 1000.0, 3).tolist()}, "
                f"command_max_mm={np.round(upper * 1000.0, 3).tolist()}, "
                f"emergency_min_mm={np.round(feedback_lower * 1000.0, 3).tolist()}, "
                f"emergency_max_mm={np.round(feedback_upper * 1000.0, 3).tolist()}"
            )

        desired_tcp = current_tcp.copy()
        absolute = self.config.action_contract == "tcp_absolute_6d_plus_o6_target_6d"
        desired_tcp[:3] = delta_tcp[:3] if absolute else current_tcp[:3] + delta_tcp[:3]
        if self.config.lock_rpy:
            if self._locked_rpy_rad is None:
                raise RuntimeError("锁定 RPY 尚未初始化")
            desired_tcp[3:6] = self._locked_rpy_rad
        else:
            # Keep the absolute ABC branch close to current feedback. Wrapping
            # the absolute target itself at +/-pi can create an artificial IK
            # branch flip even though the physical orientation is continuous.
            rpy_delta = delta_tcp[3:6] - current_tcp[3:6] if absolute else delta_tcp[3:6]
            desired_tcp[3:6] = current_tcp[3:6] + _wrap_to_pi(rpy_delta)

        # Preserve the raw reconstructed policy target for deployment logging,
        # then project only XYZ into the strict command workspace.
        self._last_raw_desired_tcp = desired_tcp.copy().astype(np.float32)
        if absolute and (np.any(desired_tcp[:3] < lower) or np.any(desired_tcp[:3] > upper)):
            raise ValueError(f"Absolute TCP target outside workspace: {desired_tcp[:3].tolist()}")
        desired_tcp[:3] = np.clip(desired_tcp[:3], lower, upper)

        # O6 action is already an absolute target in the dataset. Physical hand
        # speed/torque limiting is handled by the configured O6 motion profile.
        desired_hand = np.clip(hand_target, 0.0, 255.0).astype(np.float32)
        return command, desired_tcp.astype(np.float32), desired_hand

    def _advance_commanded_tcp(self, desired_tcp: Sequence[Any]) -> np.ndarray:
        """Rate-limit persistent commanded TCP progression toward desired_tcp."""

        desired = _finite_vector(desired_tcp, 6, "desired_tcp")
        if self._commanded_tcp is None:
            self._commanded_tcp = self.get_tcp_pose().copy()

        now = time.monotonic()
        if self._last_tcp_command_time is None:
            dt = 0.001
        else:
            dt = now - self._last_tcp_command_time
        dt = max(0.001, min(float(dt), float(self.config.tcp_tracker_max_dt_s)))

        commanded = self._commanded_tcp.copy()

        # Translation: same direction-preserving target tracker used during
        # collection: commanded += (desired-commanded) * ratio.
        remaining_xyz = desired[:3] - commanded[:3]
        distance = float(np.linalg.norm(remaining_xyz))
        max_step_m = float(self.config.max_tcp_speed_mm_s) * 1e-3 * dt
        if distance > 1e-9:
            ratio = min(1.0, max_step_m / distance)
            commanded[:3] += remaining_xyz * ratio

        # Orientation: shortest-angle branch with a common angular speed limit.
        # For the small per-cycle step (0.1 rad/s default), this preserves the
        # same target-tracking intent without forcing an absolute +/-pi wrap.
        if self.config.lock_rpy:
            if self._locked_rpy_rad is None:
                raise RuntimeError("锁定 RPY 尚未初始化")
            commanded[3:6] = self._locked_rpy_rad
        else:
            remaining_rpy = _wrap_to_pi(desired[3:6] - commanded[3:6])
            angular_distance = float(np.linalg.norm(remaining_rpy))
            max_angular_step = float(self.config.max_tcp_angular_speed_rad_s) * dt
            if angular_distance > 1e-9:
                ratio = min(1.0, max_angular_step / angular_distance)
                commanded[3:6] += remaining_rpy * ratio

        lower = np.asarray(self.config.workspace_min_m, dtype=np.float32)
        upper = np.asarray(self.config.workspace_max_m, dtype=np.float32)
        if np.any(commanded[:3] < lower) or np.any(commanded[:3] > upper):
            raise RuntimeError("限速后的 TCP command 越出工作空间")

        self._desired_tcp = desired.copy()
        self._commanded_tcp = commanded.astype(np.float32)
        self._last_tcp_command_time = now
        return self._commanded_tcp.copy()

    def _maybe_log_motion(
        self,
        *,
        current_tcp: np.ndarray,
        action: np.ndarray,
        desired_tcp: np.ndarray,
        commanded_tcp: np.ndarray,
        action_age_s: float | None,
    ) -> None:
        now = time.monotonic()
        if now - self._last_motion_log_time < float(self.config.deployment_log_period_s):
            return
        self._last_motion_log_time = now

        lower = np.asarray(self.config.workspace_min_m, dtype=np.float32)
        upper = np.asarray(self.config.workspace_max_m, dtype=np.float32)
        raw_desired = (
            self._last_raw_desired_tcp.copy()
            if self._last_raw_desired_tcp is not None
            else desired_tcp.copy()
        )
        projected = bool(np.max(np.abs(raw_desired[:3] - desired_tcp[:3])) > 1e-7)
        current_margin_min_mm = (current_tcp[:3] - lower) * 1000.0
        current_margin_max_mm = (upper - current_tcp[:3]) * 1000.0
        age_ms = None if action_age_s is None else float(action_age_s) * 1000.0

        logger.info(
            "DEPLOY TCP current_mm=%s action_xyz_mm=%s raw_desired_mm=%s "
            "desired_mm=%s commanded_mm=%s projected=%s "
            "margin_from_min_mm=%s margin_to_max_mm=%s age_ms=%s",
            np.round(current_tcp[:3] * 1000.0, 3).tolist(),
            np.round(action[:3] * 1000.0, 3).tolist(),
            np.round(raw_desired[:3] * 1000.0, 3).tolist(),
            np.round(desired_tcp[:3] * 1000.0, 3).tolist(),
            np.round(commanded_tcp[:3] * 1000.0, 3).tolist(),
            projected,
            np.round(current_margin_min_mm, 3).tolist(),
            np.round(current_margin_max_mm, 3).tolist(),
            None if age_ms is None else round(age_ms, 1),
        )
        logger.info(
            "DEPLOY RPY current_deg=%s action_rpy_deg=%s raw_desired_deg=%s "
            "desired_deg=%s commanded_deg=%s",
            np.round(np.rad2deg(current_tcp[3:6]), 3).tolist(),
            np.round(np.rad2deg(action[3:6]), 3).tolist(),
            np.round(np.rad2deg(raw_desired[3:6]), 3).tolist(),
            np.round(np.rad2deg(desired_tcp[3:6]), 3).tolist(),
            np.round(np.rad2deg(commanded_tcp[3:6]), 3).tolist(),
        )

    def solve_ik(self, target_tcp: Sequence[Any]) -> np.ndarray:
        """Convert m/rad TCP to NRC mm/rad and return seven joint degrees."""

        target = _finite_vector(target_tcp, 6, "target_tcp")
        robot = self._require_cr5()
        nrc_target = robot.tcp_position()  # preserve controller-specific seventh value
        nrc_target[:3] = (target[:3] * 1000.0).tolist()
        nrc_target[3:6] = target[3:6].tolist()
        joints_deg = np.asarray(robot.inverse_kinematics(nrc_target), dtype=np.float32)
        if joints_deg.shape != (7,) or not np.all(np.isfinite(joints_deg)):
            raise RuntimeError(f"CR5 IK 返回无效: {joints_deg}")
        return joints_deg

    def servo_j(
        self,
        target_joint_deg_7: Sequence[Any],
        *,
        deadline_ns: int | None = None,
    ) -> None:
        target = _finite_vector(target_joint_deg_7, 7, "target_joint_deg_7")
        with self._command_lock:
            if not self._control_started:
                raise RuntimeError("控制尚未启动或已停止，禁止发送 ServoJ")
            robot = self._require_cr5()
            feedback_started_ns = time.monotonic_ns()
            actual = _finite_vector(robot.joint_position(), 7, "NRC 关节反馈")
            feedback_finished_ns = time.monotonic_ns()

            # Recheck freshness after the potentially blocking feedback RPC.
            if deadline_ns is not None and feedback_finished_ns > deadline_ns:
                raise _ActionDeadlineExpired(
                    "模型动作在 ServoJ 下发前已过期: "
                    f"feedback_rpc_ms={(feedback_finished_ns - feedback_started_ns) / 1e6:.3f}, "
                    f"overdue_ms={(feedback_finished_ns - deadline_ns) / 1e6:.3f}"
                )

            # Compare unwrapped joint coordinates. A +/-360-degree IK branch
            # change must be rejected even if its TCP pose is equivalent.
            step = np.abs(np.deg2rad(target[:6] - actual[:6]))
            if self._last_joint_target_deg is not None:
                tracking = np.abs(np.deg2rad(self._last_joint_target_deg[:6] - actual[:6]))
            else:
                tracking = np.zeros(6, dtype=np.float32)

            now = time.monotonic()
            if now - self._last_joint_log_time >= float(self.config.deployment_log_period_s):
                self._last_joint_log_time = now
                logger.info(
                    "DEPLOY JOINT actual_deg=%s target_deg=%s max_target_error_deg=%.3f "
                    "max_tracking_error_deg=%.3f",
                    np.round(actual[:6], 3).tolist(),
                    np.round(target[:6], 3).tolist(),
                    float(np.max(np.rad2deg(step))),
                    float(np.max(np.rad2deg(tracking))),
                )

            if np.any(step > self.config.max_joint_step_rad):
                raise RuntimeError(f"关节下发目标与反馈距离超限(rad): {step.tolist()}")
            if np.any(tracking > self.config.max_tracking_error_rad):
                raise RuntimeError(f"关节跟踪误差超限(rad): {tracking.tolist()}")
            if not self._control_started:
                raise RuntimeError("控制已停止，取消待发送目标")
            if deadline_ns is not None and time.monotonic_ns() > deadline_ns:
                raise _ActionDeadlineExpired("模型动作在 ServoJ 下发前已过期")
            robot.send_servoj(target.tolist())
            self._last_joint_target_deg = target.copy()

    def apply_delta_tcp(self, delta_tcp: Sequence[Any]) -> np.ndarray:
        """CR5-only helper; O6 is held at its current feedback target."""

        if self.config.action_contract != "tcp_absolute_6d_plus_o6_target_6d":
            raise ValueError("apply_delta_tcp requires the residual action contract")
        delta = _finite_vector(delta_tcp, 6, "delta_tcp")
        hand = self.get_o6_positions()
        action = np.concatenate([delta, hand]).astype(np.float32)
        return self.apply_action(action)[:6]

    def apply_action(
        self, action: Sequence[Any], *, observation_timestamp_ns: int | None = None,
        _continuation: bool = False,
    ) -> np.ndarray:
        """Execute a 12D command using the explicitly configured action contract."""

        apply_started_ns = time.monotonic_ns()
        with self._command_lock:
            try:
                if not self._control_started:
                    raise RuntimeError("尚未调用 start_control()")
                robot = self._require_cr5()
                if not robot.connections_ready():
                    raise ConnectionError("CR5 6001/7000 连接已断开")
                if robot.servo_state() != 3:
                    raise RuntimeError("CR5 已离开伺服运行状态 3")
                if self.o6.error:
                    raise RuntimeError(self.o6.error)

                deadline_ns = None
                if observation_timestamp_ns is not None:
                    deadline_ns = observation_timestamp_ns + int(
                        float(self.config.action_max_age_s) * 1e9
                    )
                    age_s = (time.monotonic_ns() - observation_timestamp_ns) / 1e9
                    if age_s > self.config.action_max_age_s:
                        raise _ActionDeadlineExpired(f"模型观测已过期: {age_s:.3f}s")
                    if age_s < 0.0:
                        raise TimeoutError(f"模型观测已过期或时间戳无效: {age_s:.3f}s")

                current_tcp = self.get_tcp_pose()
                current_hand = self.get_o6_positions()
                action_array = _finite_vector(action, self.ACTION_DIM, "action")
                if self.config.action_contract == "cr3_q1_q6_absolute_plus_o6_target_6d":
                    if np.any((action_array[6:] < 0) | (action_array[6:] > 255)):
                        raise ValueError("O6 targets must be in 0..255")
                    lower = np.asarray(self.config.workspace_min_m)
                    upper = np.asarray(self.config.workspace_max_m)
                    margin = self.config.feedback_workspace_margin_m
                    if np.any(current_tcp[:3] < lower - margin) or np.any(current_tcp[:3] > upper + margin):
                        raise RuntimeError("CR3 feedback outside workspace")
                    target = _finite_vector(robot.joint_position(), 7, "NRC joint feedback")
                    now = time.monotonic()
                    dt = min(max(now - self._last_joint_command_time, 0.0), 0.05)
                    previous = np.deg2rad(self._last_joint_target_deg[:6])
                    max_step = min(self.config.joint_target_max_speed_rad_s * dt,
                                   self.config.max_joint_step_rad)
                    actual = np.deg2rad(target[:6])
                    # Limit lead over feedback before slew-limiting commands:
                    # a lagging/stationary arm must not accumulate future targets.
                    # Keep the previous accepted target intact for servo_j's
                    # independent tracking-error check against fresh feedback.
                    feedback_target = actual + np.clip(action_array[:6] - actual, -max_step, max_step)
                    commanded = previous + np.clip(feedback_target - previous, -max_step, max_step)
                    target[:6] = np.rad2deg(commanded)
                    # Preserve the controller's seventh coordinate; never run joint angles through TCP IK.
                    fk_started_ns = time.monotonic_ns()
                    target_tcp = robot.forward_kinematics(target)[:3] / 1000.0
                    fk_finished_ns = time.monotonic_ns()
                    if np.any(target_tcp < lower) or np.any(target_tcp > upper):
                        outside_axes = np.array(["x", "y", "z"])[(target_tcp < lower) | (target_tcp > upper)]
                        excess_mm = np.maximum(np.maximum(lower - target_tcp, target_tcp - upper), 0) * 1000
                        raise ValueError(
                            "CR3 joint target outside TCP workspace: "
                            f"candidate_tcp_mm={np.round(target_tcp * 1000, 3).tolist()}, "
                            f"feedback_tcp_mm={np.round(current_tcp[:3] * 1000, 3).tolist()}, "
                            f"command_min_mm={np.round(lower * 1000, 3).tolist()}, "
                            f"command_max_mm={np.round(upper * 1000, 3).tolist()}, "
                            f"outside_axes={outside_axes.tolist()}, "
                            f"excess_mm={np.round(excess_mm, 3).tolist()}, "
                            f"candidate_joint_deg={np.round(target[:6], 3).tolist()}"
                        )
                    prepared_ns = time.monotonic_ns()
                    try:
                        self.servo_j(target, deadline_ns=deadline_ns)
                    except _ActionDeadlineExpired:
                        logger.error(
                            "CR3 action deadline: entry_age_ms=%.3f prepare_ms=%.3f fk_ms=%.3f "
                            "budget_before_servoj_ms=%.3f",
                            (apply_started_ns - observation_timestamp_ns) / 1e6,
                            (prepared_ns - apply_started_ns) / 1e6,
                            (fk_finished_ns - fk_started_ns) / 1e6,
                            (deadline_ns - prepared_ns) / 1e6,
                        )
                        raise
                    accepted_hand = np.asarray(self.o6.set_positions(action_array[6:]), dtype=np.float32)
                    self._desired_hand_target = accepted_hand.copy()
                    self._desired_joint_target_rad = action_array[:6].copy()
                    self._last_joint_command_time = now
                    self._target_observation_timestamp_ns = observation_timestamp_ns
                    return np.concatenate([np.deg2rad(target[:6]), accepted_hand]).astype(np.float32)
                _, desired_tcp, desired_hand = self._filter_action(
                    action_array, current_tcp, current_hand
                )
                self._desired_tcp = desired_tcp.copy()
                self._desired_hand_target = desired_hand.copy()

                commanded_tcp = self._advance_commanded_tcp(desired_tcp)
                action_age_s = (
                    None
                    if observation_timestamp_ns is None
                    else (time.monotonic_ns() - observation_timestamp_ns) / 1e9
                )
                self._maybe_log_motion(
                    current_tcp=current_tcp,
                    action=action_array,
                    desired_tcp=desired_tcp,
                    commanded_tcp=commanded_tcp,
                    action_age_s=action_age_s,
                )
                if self.config.action_contract == "tcp_absolute_6d_plus_o6_target_6d" and self._ros_servol is not None:
                    self._ros_servol.send((commanded_tcp[:6] * np.array([1000, 1000, 1000, 1, 1, 1])).tolist())
                    target_joints_deg = None
                else:
                    target_joints_deg = self.solve_ik(commanded_tcp)

                if deadline_ns is not None and time.monotonic_ns() > deadline_ns:
                    raise _ActionDeadlineExpired("IK 完成后模型动作已过期")

                if target_joints_deg is not None:
                    self.servo_j(target_joints_deg, deadline_ns=deadline_ns)
                accepted_hand = np.asarray(
                    self.o6.set_positions(desired_hand), dtype=np.float32
                )
                self._desired_hand_target = accepted_hand.copy()

                self._target_observation_timestamp_ns = observation_timestamp_ns

                # Return the command actually accepted by this interface.
                if self.config.action_contract == "tcp_absolute_6d_plus_o6_target_6d":
                    return np.concatenate([commanded_tcp, accepted_hand]).astype(np.float32)
                executed_delta = np.empty(6, dtype=np.float32)
                executed_delta[:3] = commanded_tcp[:3] - current_tcp[:3]
                executed_delta[3:6] = _wrap_to_pi(
                    commanded_tcp[3:6] - current_tcp[3:6]
                ).astype(np.float32)
                return np.concatenate([executed_delta, accepted_hand]).astype(np.float32)
            except Exception as exc:
                if _continuation and isinstance(exc, _ActionDeadlineExpired):
                    # track_or_hold rolls back the unsent candidate and holds.
                    raise
                self._last_error = f"{type(exc).__name__}: {exc}"
                if self._control_started:
                    try:
                        self.emergency_stop()
                    except Exception as stop_error:
                        raise ExceptionGroup("动作失败且停止未完成", [exc, stop_error]) from exc
                raise

    def track_or_hold(self) -> None:
        """Advance a fresh absolute target; otherwise hold the accepted joints.

        Reuse the original observation timestamp, so ticks cannot extend the
        target lifetime. apply_action retains workspace, IK and joint checks.
        Legacy residual deployment keeps its fixed-target hold behavior.
        """
        with self._command_lock:
            stamp = self._target_observation_timestamp_ns
            if self._control_started and self.config.action_contract == "cr3_q1_q6_absolute_plus_o6_target_6d":
                if stamp is not None and 0 <= time.monotonic_ns() - stamp <= int(self.config.action_max_age_s * 1e9):
                    try:
                        self.apply_action(np.concatenate([self._desired_joint_target_rad, self._desired_hand_target]),
                                          observation_timestamp_ns=stamp, _continuation=True)
                    except _ActionDeadlineExpired:
                        self._target_observation_timestamp_ns = None
                        self.hold_position()
                else:
                    self.hold_position()
                return
            if (
                self._control_started
                and self.config.action_contract == "tcp_absolute_6d_plus_o6_target_6d"
                and stamp is not None
                and 0 <= time.monotonic_ns() - stamp <= int(self.config.action_max_age_s * 1e9)
            ):
                accepted_tcp = self._commanded_tcp.copy()
                accepted_time = self._last_tcp_command_time
                desired_tcp = self._desired_tcp.copy()
                desired_hand = self._desired_hand_target.copy()
                try:
                    self.apply_action(
                        np.concatenate([desired_tcp, desired_hand]),
                        observation_timestamp_ns=stamp,
                        _continuation=True,
                    )
                except _ActionDeadlineExpired as exc:
                    # IK/feedback may consume the remaining lifetime. No new
                    # target was sent: discard that candidate, not the session.
                    self._commanded_tcp = accepted_tcp
                    self._last_tcp_command_time = accepted_time
                    self._desired_tcp = desired_tcp
                    self._desired_hand_target = desired_hand
                    self._target_observation_timestamp_ns = None
                    logger.info("Target expired during tracking; holding accepted joints: %s", exc)
                    self.hold_position()
            else:
                self.hold_position()

    def hold_position(self) -> None:
        """Repeat the last accepted fixed joint/hand target while policy is late."""

        with self._command_lock:
            if not self._control_started:
                return
            try:
                robot = self._require_cr5()
                if not robot.connections_ready():
                    raise ConnectionError("CR5 6001/7000 连接已断开")
                if robot.servo_state() != 3:
                    raise RuntimeError("CR5 已离开伺服运行状态 3")
                if self.o6.error:
                    raise RuntimeError(self.o6.error)

                target = self._last_joint_target_deg
                if target is None:
                    target = _finite_vector(robot.joint_position(), 7, "NRC 关节反馈")
                    self._last_joint_target_deg = target.copy()
                self.servo_j(target)

                if self._desired_hand_target is not None:
                    accepted = np.asarray(
                        self.o6.set_positions(self._desired_hand_target), dtype=np.float32
                    )
                    self._desired_hand_target = accepted.copy()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                try:
                    self.emergency_stop()
                except Exception as stop_error:
                    raise ExceptionGroup("Hold failed and stop failed", [exc, stop_error]) from exc
                raise

    def hold_action(self) -> np.ndarray:
        """Return a hold in the configured physical action representation."""

        hand = self.get_o6_positions()
        if self.config.action_contract == "cr3_q1_q6_absolute_plus_o6_target_6d":
            joints = (self.get_joint_positions() if self._last_joint_target_deg is None
                      else np.deg2rad(self._last_joint_target_deg[:6]))
            return np.concatenate([joints, hand]).astype(np.float32)
        if self.config.action_contract == "tcp_absolute_6d_plus_o6_target_6d":
            tcp = self.get_tcp_pose() if self._commanded_tcp is None else self._commanded_tcp.copy()
            return np.concatenate([tcp, hand]).astype(np.float32)
        return np.concatenate([np.zeros(6, dtype=np.float32), hand]).astype(np.float32)

    def get_diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "control_started": self._control_started,
            "last_error": self._last_error,
            "o6_connected": self.o6.connected,
            "o6_phase": self.o6.phase,
            "o6_error": self.o6.error,
            "global_camera_connected": bool(
                self._global_camera is not None and self._global_camera.connected
            ),
            "wrist_camera_connected": bool(
                self._wrist_camera is not None and self._wrist_camera.connected
            ),
            "right_wrist_camera_connected": bool(
                self._right_wrist_camera is not None and self._right_wrist_camera.connected
            ),
            "tcp_tracker_initialized": self._commanded_tcp is not None,
            "max_tcp_speed_mm_s": float(self.config.max_tcp_speed_mm_s),
            "max_tcp_angular_speed_rad_s": float(self.config.max_tcp_angular_speed_rad_s),
        }
        robot = self.cr5
        if robot is None:
            result["cr5_connected"] = False
            return result
        try:
            statuses = robot.connection_statuses()
            result.update(
                {
                    "cr5_connected": all(value == 0 for value in statuses.values()),
                    "nrc_connections": statuses,
                    "servo_state": robot.servo_state(),
                    "running_state": robot.running_state(),
                    "servoj_open": robot.servoj_open,
                }
            )
        except Exception as exc:
            result.update({"cr5_connected": False, "diagnostic_error": str(exc)})
        return result

    def get_error_code(self) -> dict[str, Any]:
        """Compatibility name: NRC has no single API covering every fault source."""

        return self.get_diagnostics()

    def emergency_stop(self) -> None:
        """Best-effort software stop; it never replaces the cabinet E-stop."""

        self._control_started = False
        errors: list[str] = []
        with self._command_lock:
            robot = self.cr5
            if robot is not None:
                try:
                    robot.stop_motion()
                except Exception as exc:
                    errors.append(f"NRC 停止失败: {exc}")
            try:
                self.o6.hold_current()
            except Exception as exc:
                errors.append(f"O6 保持失败: {exc}")
        if errors:
            self._last_error = "; ".join(errors)
            raise RuntimeError(self._last_error)

    def close(self) -> None:
        """Stop command producers, close device workers, then release NRC sockets."""

        errors = []
        stop_failed = False
        if self._control_started or (self.cr5 is not None and self.cr5.servoj_open):
            try:
                self.emergency_stop()
            except Exception as exc:
                errors.append(exc)
                stop_failed = True
        if self._ros_servol is not None:
            self._ros_servol.close()
            self._ros_servol = None
        for camera in (self._global_camera, self._wrist_camera, self._right_wrist_camera):
            if camera is not None:
                try:
                    camera.close()
                except Exception as exc:
                    errors.append(exc)
        try:
            self.o6.close()
        except Exception as exc:
            errors.append(exc)

        api = self._nrc_api
        if api is not None and not stop_failed:
            for attribute in ("command_fd", "servo_fd"):
                fd = getattr(self, attribute)
                if fd > 0:
                    try:
                        api.disconnect_robot(fd)
                    except Exception as exc:
                        errors.append(exc)
                    else:
                        setattr(self, attribute, -1)
        if self.command_fd < 0 and self.servo_fd < 0:
            self.cr5 = None
            self._nrc_api = None
        if errors:
            raise ExceptionGroup("设备关闭未完成", errors)


NrcArmO6Interface = DobotCR5O6Interface

__all__ = ["DobotCR5O6Interface", "NrcArmO6Interface", "InterfaceConfig"]
