#!/usr/bin/env python3
"""
ROS1 interface wrapper for Yuanyou2 robot with agilex-level control features.

Key improvements over the basic interface:
1. BlockingDeque — thread-safe sensor buffering, decouples ROS callbacks from inference.
2. Temporal synchronization — closest-timestamp matching across all sensor streams.
3. Smooth joint interpolation — polynomial (small moves) + linear (large moves) at 200 Hz.
4. Dedicated control thread — high-frequency publishing independent of inference rate.
5. Preemptive publishing — new commands interrupt in-progress interpolation.
6. JPEG compression/decompression pipeline — matches training data distribution.
7. Async inference + RTC guidance support.
8. Image preprocessing — center-crop/pad + resize to 224x224.

Responsibilities:
1. Subscribe to three camera topics.
2. Subscribe to /joint_states.
3. Build 14-dim Yuanyou2 state:
   [left_arm_6, left_gripper, right_arm_6, right_gripper]
4. Publish 14-dim Yuanyou2 action to real robot command topics with interpolation.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState


# ---------------------------------------------------------------------------
# BlockingDeque — thread-safe double-ended queue with condition variable.
# Ported from agilex openpi-agilex.
# ---------------------------------------------------------------------------

class BlockingDeque:
    """Thread-safe deque that supports blocking pop operations."""

    def __init__(self, maxlen: int = 200):
        self._deque: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def append(self, item):
        with self._cond:
            self._deque.append(item)
            self._cond.notify_all()

    def popleft(self):
        with self._cond:
            return self._deque.popleft()

    def pop(self):
        with self._cond:
            return self._deque.pop()

    def __len__(self):
        with self._lock:
            return len(self._deque)

    def peek_last(self):
        with self._lock:
            if self._deque:
                return self._deque[-1]
            return None

    def clear(self):
        with self._lock:
            self._deque.clear()

    def get_snapshot(self) -> list:
        with self._lock:
            return list(self._deque)


# ---------------------------------------------------------------------------
# Image preprocessing helpers — ported from agilex openpi-agilex.
# ---------------------------------------------------------------------------

def _center_crop_or_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop or pad an image to exactly (target_h, target_w)."""
    h, w = image.shape[:2]

    if h > target_h:
        start = (h - target_h) // 2
        image = image[start:start + target_h, :]
    elif h < target_h:
        pad = (target_h - h) // 2
        image = np.pad(image, ((pad, target_h - h - pad), (0, 0), (0, 0)), mode="constant")

    if w > target_w:
        start = (w - target_w) // 2
        image = image[:, start:start + target_w]
    elif w < target_w:
        pad = (target_w - w) // 2
        image = np.pad(image, ((0, 0), (pad, target_w - w - pad), (0, 0)), mode="constant")

    return image


def resize_with_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image keeping aspect ratio, then center-pad to target size."""
    h, w = image.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return _center_crop_or_pad(resized, target_h, target_w)


def jpeg_compress_decompress(image: np.ndarray, quality: int = 95) -> np.ndarray:
    """Simulate JPEG compression pipeline to match training data distribution."""
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Yuanyou2ROS1Interface
# ---------------------------------------------------------------------------

class Yuanyou2ROS1Interface:
    """Thread-safe ROS1 interface for Yuanyou2 dual-arm robot.

    Features ported from agilex openpi-agilex:
    - BlockingDeque sensor buffering
    - Temporal synchronization across cameras and joint states
    - Smooth joint interpolation at 200 Hz with polynomial + linear modes
    - Preemptive publishing (new command cancels in-progress motion)
    - JPEG compression/decompression pipeline
    - Image preprocessing (center-crop + resize 224x224)
    """

    # Yuanyou2 14-dim layout: [left_arm_6, left_gripper, right_arm_6, right_gripper]
    STATE_DIM = 14
    ACTION_DIM = 14

    # Per-step position/gripper tolerance for interpolation step count.
    DEFAULT_ARM_STEPS_LENGTH = [0.005] * 6 + [0.01]  # 6 joints + gripper

    def __init__(
        self,
        node_name: str = "yuanyou2_openpi_interface",
        image_topics: dict[str, str] | None = None,
        # Interpolation / control parameters
        interpolation_hz: float = 200.0,
        arm_steps_length: list[float] | None = None,
        preemptive_publishing: bool = True,
        # Image preprocessing
        use_jpeg_pipeline: bool = True,
        jpeg_quality: int = 95,
        render_h: int = 224,
        render_w: int = 224,
        obs_history_num: int = 1,
        # Action chunking
        chunk_size: int = 50,
        # Asynchronous inference
        asynchronous_inference: bool = False,
        # RTC guidance
        use_rtc_guidance: bool = False,
    ):
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        self.logger = logging.getLogger(__name__)
        self.cv_bridge = CvBridge()

        # ---- sensor buffers (BlockingDeque) ----
        self._image_queues: Dict[str, BlockingDeque] = {}
        self._joint_state_queue: BlockingDeque = BlockingDeque(maxlen=200)

        # ---- interpolation / control ----
        self._interpolation_hz = interpolation_hz
        self._arm_steps_length = arm_steps_length or self.DEFAULT_ARM_STEPS_LENGTH
        self._preemptive_publishing = preemptive_publishing

        # ---- image preprocessing ----
        self._use_jpeg_pipeline = use_jpeg_pipeline
        self._jpeg_quality = jpeg_quality
        self._render_h = render_h
        self._render_w = render_w
        self._obs_history_num = obs_history_num

        # ---- action chunking ----
        self._chunk_size = chunk_size

        # ---- async inference ----
        self._asynchronous_inference = asynchronous_inference

        # ---- RTC guidance ----
        self._use_rtc_guidance = use_rtc_guidance

        # ---- state ----
        # Current action chunk being executed.
        self._current_chunk: np.ndarray | None = None
        self._chunk_start_time: float = 0.0
        self._chunk_step: int = 0
        self._target_reached: bool = True
        self._chunk_lock = threading.Lock()

        # Last commanded joint positions for interpolation start point.
        self._last_command_left: np.ndarray | None = None
        self._last_command_right: np.ndarray | None = None

        # History for polynomial interpolation (k=3).
        self._joint_history_left: list[np.ndarray] = []
        self._joint_history_right: list[np.ndarray] = []
        self._poly_k = 3

        # Previous action chunk for RTC guidance.
        self._prev_action_chunk: np.ndarray | None = None
        self._prev_executed_steps: int = 0

        # Threading: control thread running at interpolation_hz.
        self._ctrl_thread: threading.Thread | None = None
        self._ctrl_running = False

        self._warn_if_system_time_unset()

        # ---- topics ----
        default_image_topics = {
            "head": "/head_camera/usb_cam/image_raw",
            "left_wrist": "/left_wrist_d435/color/image_raw",
            "right_wrist": "/right_wrist_d435/color/image_raw",
        }
        configured_image_topics = dict(default_image_topics)
        if image_topics:
            configured_image_topics.update(image_topics)

        self.image_topics = {
            "head": rospy.get_param("~head_image_topic", configured_image_topics["head"]),
            "left_wrist": rospy.get_param("~left_wrist_image_topic", configured_image_topics["left_wrist"]),
            "right_wrist": rospy.get_param("~right_wrist_image_topic", configured_image_topics["right_wrist"]),
        }
        self.joint_state_topic = rospy.get_param("~joint_state_topic", "/joint_states")
        self.left_cmd_topic = rospy.get_param("~left_cmd_topic", "/left/joint_cmd")
        self.right_cmd_topic = rospy.get_param("~right_cmd_topic", "/right/joint_cmd")

        # ---- gripper params ----
        self.use_gripper = bool(rospy.get_param("~use_gripper", True))
        self.clamp_gripper = bool(rospy.get_param("~clamp_gripper", True))
        self.gripper_min = float(rospy.get_param("~gripper_min", 0.0))
        self.gripper_max = float(rospy.get_param("~gripper_max", 0.035))
        self.arm_velocity_default = float(rospy.get_param("~arm_velocity_default", 0.0))
        self.arm_effort_default = float(rospy.get_param("~arm_effort_default", 0.0))
        self.gripper_velocity_default = float(rospy.get_param("~gripper_velocity_default", 10.0))
        self.gripper_effort_default = float(rospy.get_param("~gripper_effort_default", 0.5))

        # ---- joint names ----
        self.left_arm_joints = [
            "left_joint1", "left_joint2", "left_joint3",
            "left_joint4", "left_joint5", "left_joint6",
        ]
        self.right_arm_joints = [
            "right_joint1", "right_joint2", "right_joint3",
            "right_joint4", "right_joint5", "right_joint6",
        ]
        self.left_gripper_joint = "left_joint7"
        self.right_gripper_joint = "right_joint7"

        # ---- setup ----
        self._setup_subscribers()
        self._setup_publishers()
        self._start_control_thread()

        self.logger.info("Yuanyou2ROS1Interface initialized (agilex-enhanced).")
        self.logger.info("image_topics=%s", self.image_topics)
        self.logger.info("interpolation_hz=%.1f, preemptive=%s, jpeg_pipeline=%s, "
                         "obs_history=%d, chunk_size=%d",
                         interpolation_hz, preemptive_publishing,
                         use_jpeg_pipeline, obs_history_num, chunk_size)

    # ------------------------------------------------------------------
    # System clock check
    # ------------------------------------------------------------------

    def _warn_if_system_time_unset(self):
        if time.time() < 1577836800:
            self.logger.warning(
                "System clock appears unset. Fix NTP/date before collecting serious rosbag data."
            )

    # ------------------------------------------------------------------
    # ROS subscribers / publishers
    # ------------------------------------------------------------------

    def _setup_subscribers(self):
        for camera_name in self.image_topics:
            queue: BlockingDeque = BlockingDeque(maxlen=200)
            self._image_queues[camera_name] = queue
            rospy.Subscriber(
                self.image_topics[camera_name],
                Image,
                lambda msg, name=camera_name: self._image_callback(msg, name),
                queue_size=1,
            )

        rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self._joint_state_callback,
            queue_size=1,
        )

        self.logger.info("ROS1 subscribers initialized (%d cameras + joint_states).",
                         len(self._image_queues))

    def _setup_publishers(self):
        self.left_cmd_pub = rospy.Publisher(
            self.left_cmd_topic, JointState, queue_size=1,
        )
        self.right_cmd_pub = rospy.Publisher(
            self.right_cmd_topic, JointState, queue_size=1,
        )
        self.logger.info("ROS1 publishers initialized.")

    def _image_callback(self, msg: Image, camera_name: str):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            timestamp = msg.header.stamp.to_sec()
            self._image_queues[camera_name].append((timestamp, image))
        except Exception as exc:
            self.logger.error("Failed to process %s image: %s", camera_name, exc)

    def _joint_state_callback(self, msg: JointState):
        timestamp = msg.header.stamp.to_sec()
        self._joint_state_queue.append((timestamp, msg))

    # ------------------------------------------------------------------
    # Initial data check
    # ------------------------------------------------------------------

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        required_images = list(self.image_topics.keys())
        start_time = time.time()

        while time.time() - start_time < timeout:
            images_ready = all(len(q) > 0 for q in self._image_queues.values())
            joints_ready = len(self._joint_state_queue) > 0

            if images_ready and joints_ready:
                self.logger.info("All Yuanyou2 sensor data received.")
                return True

            time.sleep(0.1)

        missing_images = [k for k in required_images if len(self._image_queues[k]) == 0]
        joints_ready = len(self._joint_state_queue) > 0
        self.logger.error(
            "Timeout waiting for sensor data. missing_images=%s, joints_ready=%s",
            missing_images, joints_ready,
        )
        self._log_published_topic_hint()
        return False

    def _log_published_topic_hint(self):
        try:
            published_topics = {name for name, _type in rospy.get_published_topics()}
        except Exception as exc:
            self.logger.warning("Could not query published topics: %s", exc)
            return

        expected_topics = [self.joint_state_topic] + list(self.image_topics.values())
        missing = [t for t in expected_topics if t not in published_topics]
        if missing:
            self.logger.error("Expected topics not currently published: %s", missing)

    # ------------------------------------------------------------------
    # Temporal synchronization — ported from agilex get_frame()
    # ------------------------------------------------------------------

    def _get_closest_msg(self, queue: BlockingDeque, ref_time: float) -> tuple | None:
        """Find and return the item in queue closest to ref_time, consuming stale items.

        Pops all items older than the best match, keeping the queue clean.
        Returns (timestamp, data) tuple or None if queue is empty.
        """
        snapshot = queue.get_snapshot()
        if not snapshot:
            return None

        # Find the item closest to ref_time.
        best_idx = 0
        best_diff = float("inf")
        for i, item in enumerate(snapshot):
            diff = abs(item[0] - ref_time)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        # Pop all items up to and including best_idx.
        best_item = None
        for _ in range(best_idx + 1):
            if len(queue) > 0:
                best_item = queue.popleft()

        return best_item

    def get_frame(self) -> dict | None:
        """Synchronize all sensor streams via closest-timestamp matching.

        Returns a dict with:
            "images": {cam_name: np.ndarray, ...}
            "joint_state": JointState msg
            "timestamp": float (reference frame time)
        or None if any sensor is missing.
        """
        # Find the latest timestamp across all sensors as reference.
        ref_time = 0.0
        for q in self._image_queues.values():
            last = q.peek_last()
            if last is not None:
                ref_time = max(ref_time, last[0])
        last_joint = self._joint_state_queue.peek_last()
        if last_joint is not None:
            ref_time = max(ref_time, last_joint[0])

        if ref_time == 0.0:
            return None

        # Extract closest-match data for each sensor.
        images = {}
        for cam_name, queue in self._image_queues.items():
            result = self._get_closest_msg(queue, ref_time)
            if result is None:
                return None
            images[cam_name] = result[1]

        joint_result = self._get_closest_msg(self._joint_state_queue, ref_time)
        if joint_result is None:
            return None

        return {
            "images": images,
            "joint_state": joint_result[1],
            "timestamp": ref_time,
        }

    # ------------------------------------------------------------------
    # Image preprocessing — ported from agilex get_camera_color()
    # ------------------------------------------------------------------

    def _process_camera_image(self, image: np.ndarray) -> np.ndarray:
        """Center-crop/pad to 640x480, JPEG pipeline, resize to 224x224."""
        # Step 1: center-crop or pad to 640x480 (training data format).
        processed = _center_crop_or_pad(image, 480, 640)

        # Step 2: JPEG compression/decompression to match training data distribution.
        if self._use_jpeg_pipeline:
            processed = jpeg_compress_decompress(processed, self._jpeg_quality)

        # Step 3: resize with pad to model input size (224x224).
        processed = resize_with_pad(processed, self._render_h, self._render_w)

        return processed

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _extract_joint_position(self, msg: JointState, joint_name: str) -> float:
        if joint_name not in msg.name:
            raise ValueError(
                f"Joint {joint_name} not found in /joint_states. "
                f"Available joints: {list(msg.name)}"
            )
        idx = msg.name.index(joint_name)
        if idx >= len(msg.position):
            raise ValueError(f"Joint {joint_name} has no position value.")
        return float(msg.position[idx])

    def _extract_state_14d(self, msg: JointState) -> np.ndarray:
        left_arm = [self._extract_joint_position(msg, name) for name in self.left_arm_joints]
        left_gripper = self._extract_joint_position(msg, self.left_gripper_joint)
        right_arm = [self._extract_joint_position(msg, name) for name in self.right_arm_joints]
        right_gripper = self._extract_joint_position(msg, self.right_gripper_joint)

        state = np.asarray(left_arm + [left_gripper] + right_arm + [right_gripper], dtype=np.float32)
        if state.shape != (self.STATE_DIM,):
            raise ValueError(f"Expected state shape ({self.STATE_DIM},), got {state.shape}")
        return state

    # ------------------------------------------------------------------
    # Observation building (for policy inference)
    # ------------------------------------------------------------------

    def get_observation(self) -> Optional[Dict]:
        """Build a synchronized, preprocessed observation for the policy.

        Returns a dict ready for openpi_client consumption, or None if data missing.
        """
        frame = self.get_frame()
        if frame is None:
            return None

        # Process images.
        processed_images = {}
        for cam_name, image in frame["images"].items():
            processed_images[cam_name] = self._process_camera_image(image)

        # Extract 14-dim state.
        state_14d = self._extract_state_14d(frame["joint_state"])

        return {
            "images": processed_images,
            "state": state_14d,
        }

    def get_raw_observation(self) -> Optional[Dict]:
        """Get observation without JPEG pipeline (for data collection / debugging)."""
        frame = self.get_frame()
        if frame is None:
            return None

        return {
            "images": {k: v.copy() for k, v in frame["images"].items()},
            "state": self._extract_state_14d(frame["joint_state"]),
        }

    # ------------------------------------------------------------------
    # Action publishing — with interpolation via control thread
    # ------------------------------------------------------------------

    def publish_action(self, actions: np.ndarray):
        """Queue an action for interpolated execution.

        Args:
            actions: np.ndarray of shape (ACTION_DIM,) — single 14-dim action,
                     or (chunk_size, ACTION_DIM) — a full action chunk (only first
                     action used as immediate target; subsequent steps are consumed
                     by the control thread at the policy rate).
        """
        actions = np.asarray(actions, dtype=np.float32)

        if actions.ndim == 1:
            if actions.shape != (self.ACTION_DIM,):
                self.logger.error("Expected (%d,)-dim action, got %s", self.ACTION_DIM, actions.shape)
                return
            actions = actions[np.newaxis, :]

        if actions.ndim != 2 or actions.shape[1] != self.ACTION_DIM:
            self.logger.error("Expected (N, %d) action chunk, got %s", self.ACTION_DIM, actions.shape)
            return

        with self._chunk_lock:
            # Store previous chunk for RTC guidance.
            if self._use_rtc_guidance and self._current_chunk is not None:
                executed = self._chunk_step
                self._prev_action_chunk = self._current_chunk.copy()
                self._prev_executed_steps = executed

            self._current_chunk = actions.copy()
            self._chunk_start_time = time.time()
            self._chunk_step = 0
            # Reset interpolation target reached flag so we start moving immediately.
            self._target_reached = False

    def get_prev_action_chunk(self) -> np.ndarray | None:
        """Return the previous action chunk for RTC guidance."""
        return self._prev_action_chunk

    def get_prev_executed_steps(self) -> int:
        """Return the number of steps executed from the previous chunk."""
        return self._prev_executed_steps

    # ------------------------------------------------------------------
    # Control thread — high-frequency interpolated publishing
    # ------------------------------------------------------------------

    def _start_control_thread(self):
        self._ctrl_running = True
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()
        self.logger.info("Control thread started at %.0f Hz.", self._interpolation_hz)

    def stop(self):
        """Stop the control thread."""
        self._ctrl_running = False
        if self._ctrl_thread is not None:
            self._ctrl_thread.join(timeout=2.0)
        self.logger.info("Control thread stopped.")

    def _control_loop(self):
        """High-frequency loop that publishes interpolated joint commands.

        Each iteration reads the current target action from the chunk,
        computes one interpolation step toward it, and publishes.
        The chunk step is advanced once the target is reached.
        """
        rate = rospy.Rate(self._interpolation_hz)

        while self._ctrl_running and not rospy.is_shutdown():
            self._publish_interpolated_step()
            rate.sleep()

    def _publish_interpolated_step(self):
        """Compute and publish one interpolated joint command step."""
        with self._chunk_lock:
            chunk = self._current_chunk
            step = self._chunk_step

        if chunk is None or step >= chunk.shape[0]:
            # No chunk or chunk exhausted — hold last position.
            if self._last_command_left is not None and self._last_command_right is not None:
                self._publish_joint_cmd(self._last_command_left, self._last_command_right)
            return

        target = chunk[step].copy()
        target_left = target[0:7]
        target_right = target[7:14]

        # Get current interpolated position.
        current_left, current_right = self._get_current_joint_positions()

        # Compute one interpolation step toward the target.
        left_cmd = self._interpolate_step(current_left, target_left, self._joint_history_left)
        right_cmd = self._interpolate_step(current_right, target_right, self._joint_history_right)

        # Clamp gripper.
        if self.clamp_gripper:
            left_cmd[6] = float(np.clip(left_cmd[6], self.gripper_min, self.gripper_max))
            right_cmd[6] = float(np.clip(right_cmd[6], self.gripper_min, self.gripper_max))

        # Publish.
        self._publish_joint_cmd(left_cmd, right_cmd)

        # Update history for polynomial interpolation.
        self._joint_history_left.append(left_cmd.copy())
        self._joint_history_right.append(right_cmd.copy())
        if len(self._joint_history_left) > self._poly_k:
            self._joint_history_left.pop(0)
        if len(self._joint_history_right) > self._poly_k:
            self._joint_history_right.pop(0)

        self._last_command_left = left_cmd
        self._last_command_right = right_cmd

        # Check if target reached — if so, advance chunk step.
        dist_left = np.max(np.abs(left_cmd - target_left))
        dist_right = np.max(np.abs(right_cmd - target_right))
        tolerance = 0.001  # ~0.06 degrees for joints

        if dist_left < tolerance and dist_right < tolerance:
            with self._chunk_lock:
                self._chunk_step += 1

    def _publish_joint_cmd(self, left_cmd: np.ndarray, right_cmd: np.ndarray):
        """Build and publish joint command messages."""
        now = rospy.Time.now()
        left_msg = self._build_cmd_msg(now, self.left_arm_joints, self.left_gripper_joint,
                                       left_cmd[:6], float(left_cmd[6]))
        right_msg = self._build_cmd_msg(now, self.right_arm_joints, self.right_gripper_joint,
                                        right_cmd[:6], float(right_cmd[6]))
        self.left_cmd_pub.publish(left_msg)
        self.right_cmd_pub.publish(right_msg)

    def _get_current_joint_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current joint positions from last command or latest joint state."""
        # Use last commanded position if available, otherwise read from joint state.
        if self._last_command_left is not None and self._last_command_right is not None:
            return self._last_command_left.copy(), self._last_command_right.copy()

        # Fall back to latest joint state.
        last = self._joint_state_queue.peek_last()
        if last is not None:
            state = self._extract_state_14d(last[1])
            return state[0:7].copy(), state[7:14].copy()

        # Ultimate fallback: zeros.
        return np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)

    def _interpolate_step(
        self,
        current: np.ndarray,
        target: np.ndarray,
        history: list[np.ndarray],
    ) -> np.ndarray:
        """Compute one interpolation step toward target.

        For large deltas (> max_step), moves at most max_step toward target (linear).
        For medium deltas (0.5 ~ max_step), uses polynomial interpolation through history.
        For small deltas (< tolerance), snaps directly to target.

        Called at 200 Hz, so each step moves a small fraction of the total path.
        """
        current = np.asarray(current, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        delta = target - current
        max_delta = float(np.max(np.abs(delta)))

        # Per-joint step size: use the joint step (0.005 rad ≈ 0.3 deg per step).
        joint_step = self._arm_steps_length[0]  # 0.005 rad
        gripper_step = self._arm_steps_length[-1]  # 0.01

        if max_delta < 0.001:  # ~0.06 degrees — close enough.
            return target.copy().astype(np.float32)

        if max_delta > 0.5:
            # Large motion: linear step capped to max_step per axis.
            # Scale delta so no axis exceeds its per-step limit.
            step_limits = np.array(self._arm_steps_length, dtype=np.float64)
            scale = 1.0
            for i in range(len(delta)):
                if abs(delta[i]) > step_limits[i]:
                    scale = min(scale, step_limits[i] / abs(delta[i]))
            result = current + scale * delta
            return result.astype(np.float32)

        # Medium motion: polynomial interpolation toward target.
        return self._polynomial_interpolate_step(current, target, history)

    def _polynomial_interpolate_step(
        self, current: np.ndarray, target: np.ndarray, history: list[np.ndarray]
    ) -> np.ndarray:
        """Compute next step using polynomial extrapolation through history.

        Fits a degree-(k-1) polynomial through the last k-1 commands + current position,
        then extrapolates one step forward, blending with the target to ensure convergence.
        """
        k = self._poly_k
        current = np.asarray(current, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        n_joints = len(target)
        result = np.zeros(n_joints, dtype=np.float64)

        # Build history: last k-1 commands + current as the latest point.
        hist = list(history[-k+1:] if len(history) >= k-1 else history[:])
        points = hist + [current]

        if len(points) >= 2:
            for j in range(n_joints):
                y = np.array([p[j] for p in points] + [target[j]], dtype=np.float64)
                x = np.arange(len(y), dtype=np.float64)
                # Fit poly through points + target, evaluate one step beyond current.
                coeffs = np.polyfit(x, y, deg=min(len(y) - 1, k))
                extrapolated = np.polyval(coeffs, len(y))
                # Blend: 70% toward target, 30% polynomial extrapolation.
                direct_step = current[j] + 0.3 * (target[j] - current[j])
                result[j] = 0.7 * extrapolated + 0.3 * direct_step
        else:
            # Not enough history: simple linear step.
            result = current + 0.3 * (target - current)

        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Build JointState message
    # ------------------------------------------------------------------

    def _build_cmd_msg(
        self,
        stamp: rospy.Time,
        arm_joints: list[str],
        gripper_joint: str,
        arm_positions: np.ndarray,
        gripper_position: float,
    ) -> JointState:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(arm_joints)
        msg.position = list(np.asarray(arm_positions, dtype=np.float32))
        msg.velocity = [self.arm_velocity_default] * len(arm_joints)
        msg.effort = [self.arm_effort_default] * len(arm_joints)

        if self.use_gripper:
            msg.name.append(gripper_joint)
            msg.position.append(float(gripper_position))
            msg.velocity.append(self.gripper_velocity_default)
            msg.effort.append(self.gripper_effort_default)

        return msg
