#!/usr/bin/env python3
"""Preflight checks before Yuanyou2 data collection or policy rollout.

Validates:
- ROS master connectivity
- All required image topics are publishing
- Camera info topics (intrinsics) are available
- Joint state topic has all 14 required joints
- System clock is set correctly
- Image resolution and encoding sanity check
- Topic publishing rates (hz)
"""

import argparse
import sys
import time

import rospy
from sensor_msgs.msg import CameraInfo, Image, JointState


IMAGE_TOPICS = {
    "head": "/head_camera/usb_cam/image_raw",
    "left_wrist": "/left_wrist_d435/color/image_raw",
    "right_wrist": "/right_wrist_d435/color/image_raw",
}

CAMERA_INFO_TOPICS = {
    "head": "/head_camera/usb_cam/camera_info",
    "left_wrist": "/left_wrist_d435/color/camera_info",
    "right_wrist": "/right_wrist_d435/color/camera_info",
}

REQUIRED_JOINTS = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
]

# Minimum expected publishing rates (Hz).
MIN_RATES = {
    "joint_states": 10.0,
    "image": 5.0,
}


def wait_msg(topic, msg_type, timeout):
    try:
        return rospy.wait_for_message(topic, msg_type, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"{topic}: {exc}") from exc


def measure_rate(topic, msg_type, duration: float = 3.0) -> float:
    """Measure the publishing rate of a topic over `duration` seconds."""
    start = time.time()
    count = 0
    while time.time() - start < duration:
        try:
            rospy.wait_for_message(topic, msg_type, timeout=1.0)
            count += 1
        except Exception:
            break
    elapsed = time.time() - start
    return count / elapsed if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Yuanyou2 OpenPI ROS preflight check (agilex-enhanced)"
    )
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout per topic check (seconds)")
    parser.add_argument("--check-rates", action="store_true",
                        help="Measure topic publishing rates (takes a few seconds)")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--head-image-topic", default=IMAGE_TOPICS["head"])
    parser.add_argument("--left-wrist-image-topic", default=IMAGE_TOPICS["left_wrist"])
    parser.add_argument("--right-wrist-image-topic", default=IMAGE_TOPICS["right_wrist"])
    parser.add_argument("--head-camera-info-topic", default=CAMERA_INFO_TOPICS["head"])
    parser.add_argument("--left-wrist-camera-info-topic", default=CAMERA_INFO_TOPICS["left_wrist"])
    parser.add_argument("--right-wrist-camera-info-topic", default=CAMERA_INFO_TOPICS["right_wrist"])
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("yuanyou2_openpi_preflight", anonymous=True)

    failures = []
    warnings = []

    # ---- System clock ----
    if time.time() < 1577836800:
        msg = "System clock appears unset; fix Jetson date/NTP before serious bag collection."
        failures.append(msg)
        print(f"[WARN] {msg}")

    # ---- ROS master ----
    try:
        rospy.get_master().getPid()
        print("[OK] ROS master is reachable.")
    except Exception as exc:
        failures.append(f"ROS master unreachable: {exc}")
        print(f"[FAIL] ROS master unreachable: {exc}")

    # ---- Joint states ----
    try:
        joint_msg = wait_msg(args.joint_state_topic, JointState, args.timeout)
        missing = [joint for joint in REQUIRED_JOINTS if joint not in joint_msg.name]
        if missing:
            failures.append(f"Missing joints: {missing}; available={list(joint_msg.name)}")
            print(f"[FAIL] {args.joint_state_topic}: missing joints={missing}")
        else:
            print(f"[OK] {args.joint_state_topic}: {len(joint_msg.name)} joints, "
                  f"all {len(REQUIRED_JOINTS)} required joints present")
    except Exception as exc:
        failures.append(str(exc))
        print(f"[FAIL] {args.joint_state_topic}: {exc}")

    # ---- Image topics ----
    image_topics = {
        "head": args.head_image_topic,
        "left_wrist": args.left_wrist_image_topic,
        "right_wrist": args.right_wrist_image_topic,
    }

    for name, topic in image_topics.items():
        try:
            msg = wait_msg(topic, Image, args.timeout)
            # Sanity check encoding and resolution.
            encoding_ok = msg.encoding in ("rgb8", "bgr8", "bayer_rggb8", "mono8", "16UC1")
            res_ok = msg.width > 0 and msg.height > 0
            status = []
            if not encoding_ok:
                status.append(f"unusual encoding={msg.encoding}")
            if not res_ok:
                status.append(f"bad resolution={msg.width}x{msg.height}")
            if status:
                warnings.append(f"{name} image {topic}: {', '.join(status)}")
                print(f"[WARN] {name} image {topic}: {msg.width}x{msg.height} "
                      f"enc={msg.encoding} frame={msg.header.frame_id} ({'; '.join(status)})")
            else:
                print(f"[OK] {name} image {topic}: {msg.width}x{msg.height} "
                      f"enc={msg.encoding} frame={msg.header.frame_id}")
        except Exception as exc:
            failures.append(str(exc))
            print(f"[FAIL] {name} image {topic}: {exc}")

    # ---- Camera info topics ----
    camera_info_topics = {
        "head": args.head_camera_info_topic,
        "left_wrist": args.left_wrist_camera_info_topic,
        "right_wrist": args.right_wrist_camera_info_topic,
    }

    for name, topic in camera_info_topics.items():
        try:
            msg = wait_msg(topic, CameraInfo, args.timeout)
            # Sanity check intrinsics.
            fx_ok = msg.K[0] > 0
            fy_ok = msg.K[4] > 0
            if not (fx_ok and fy_ok):
                warnings.append(f"{name} camera_info: suspicious intrinsics K={msg.K[:5]}")
                print(f"[WARN] {name} camera_info {topic}: "
                      f"frame={msg.header.frame_id} K[0]={msg.K[0]:.1f} K[4]={msg.K[4]:.1f}")
            else:
                print(f"[OK] {name} camera_info {topic}: "
                      f"frame={msg.header.frame_id} fx={msg.K[0]:.1f} fy={msg.K[4]:.1f}")
        except Exception as exc:
            # Camera info is non-critical for inference (only needed for data collection).
            warnings.append(f"{name} camera_info {topic}: {exc}")
            print(f"[WARN] {name} camera_info {topic}: {exc}")

    # ---- Publishing rates (optional) ----
    if args.check_rates:
        print("\n--- Measuring publishing rates ---")
        rate = measure_rate(args.joint_state_topic, JointState)
        if rate < MIN_RATES["joint_states"]:
            warnings.append(f"Joint state rate {rate:.1f} Hz < {MIN_RATES['joint_states']} Hz")
            print(f"[WARN] {args.joint_state_topic}: {rate:.1f} Hz (min {MIN_RATES['joint_states']} Hz)")
        else:
            print(f"[OK] {args.joint_state_topic}: {rate:.1f} Hz")

        for name, topic in image_topics.items():
            rate = measure_rate(topic, Image)
            if rate < MIN_RATES["image"]:
                warnings.append(f"{name} image rate {rate:.1f} Hz < {MIN_RATES['image']} Hz")
                print(f"[WARN] {name} {topic}: {rate:.1f} Hz (min {MIN_RATES['image']} Hz)")
            else:
                print(f"[OK] {name} {topic}: {rate:.1f} Hz")

    # ---- Summary ----
    print(f"\n{'='*50}")
    if failures:
        print(f"[FAIL] Yuanyou2 OpenPI preflight found {len(failures)} error(s):")
        for f in failures:
            print(f"  - {f}")
    if warnings:
        print(f"[WARN] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    if not failures and not warnings:
        print("[PASS] Yuanyou2 OpenPI ROS topics look ready.")
    elif not failures:
        print("[PASS] All critical checks passed (with warnings).")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
