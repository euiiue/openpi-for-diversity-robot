import argparse
import shutil
from pathlib import Path
from bisect import bisect_left

import cv2
import numpy as np


IMAGE_TOPICS = {
    "head": "/head_camera/usb_cam/image_raw",
    "left_wrist": "/left_wrist_d435/color/image_raw",
    "right_wrist": "/right_wrist_d435/color/image_raw",
}

JOINT_STATE_TOPIC = "/joint_states"
CMD_TOPICS = {
    "left": "/left/joint_cmd",
    "right": "/right/joint_cmd",
}

# 14D Yuanyou2 layout. joint8 is the coupled opposite finger in the URDF.
LEFT_ARM_JOINTS = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
]

RIGHT_ARM_JOINTS = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
]

LEFT_GRIPPER_JOINT = "left_joint7"
RIGHT_GRIPPER_JOINT = "right_joint7"

YUANYOU2_MOTOR_NAMES = (
    LEFT_ARM_JOINTS
    + ["left_gripper"]
    + RIGHT_ARM_JOINTS
    + ["right_gripper"]
)


def get_stamp_sec(msg, fallback_time):
    if hasattr(msg, "header") and msg.header.stamp:
        return msg.header.stamp.to_sec()
    return fallback_time.to_sec()


def resize_rgb(image, size=(224, 224)):
    image = cv2.resize(image, size)
    return image.astype(np.uint8)


def nearest_timed_item(items, target_time, max_delta=None):
    """
    items: list of (timestamp, data), sorted by timestamp
    """
    if not items:
        return None

    times = [x[0] for x in items]
    idx = bisect_left(times, target_time)

    if idx <= 0:
        candidate = items[0]
        return candidate if max_delta is None or abs(candidate[0] - target_time) <= max_delta else None
    if idx >= len(items):
        candidate = items[-1]
        return candidate if max_delta is None or abs(candidate[0] - target_time) <= max_delta else None

    before = items[idx - 1]
    after = items[idx]

    if abs(before[0] - target_time) <= abs(after[0] - target_time):
        candidate = before
    else:
        candidate = after

    return candidate if max_delta is None or abs(candidate[0] - target_time) <= max_delta else None


def nearest_item(items, target_time, max_delta=None):
    item = nearest_timed_item(items, target_time, max_delta=max_delta)
    return None if item is None else item[1]


def extract_joint_position(msg, joint_name):
    if joint_name not in msg.name:
        raise ValueError(f"Joint {joint_name} not found in /joint_states. Names: {msg.name}")

    idx = msg.name.index(joint_name)

    if len(msg.position) <= idx:
        raise ValueError(f"Joint {joint_name} has no position value.")

    return float(msg.position[idx])


def extract_state_14d(joint_msg):
    left_arm = [extract_joint_position(joint_msg, name) for name in LEFT_ARM_JOINTS]
    left_gripper = extract_joint_position(joint_msg, LEFT_GRIPPER_JOINT)

    right_arm = [extract_joint_position(joint_msg, name) for name in RIGHT_ARM_JOINTS]
    right_gripper = extract_joint_position(joint_msg, RIGHT_GRIPPER_JOINT)

    state = np.array(
        left_arm + [left_gripper] + right_arm + [right_gripper],
        dtype=np.float32,
    )

    if state.shape != (14,):
        raise ValueError(f"Expected state shape (14,), got {state.shape}")

    return state


def extract_command_7d(msg, arm_joints, gripper_joint, fallback_gripper):
    if msg is None:
        return None

    if len(msg.name) == len(msg.position) and msg.name:
        name_to_pos = {name: float(position) for name, position in zip(msg.name, msg.position)}
        if all(name in name_to_pos for name in arm_joints):
            arm = [name_to_pos[name] for name in arm_joints]
            gripper = name_to_pos.get(gripper_joint, fallback_gripper)
            return np.asarray(arm + [float(gripper)], dtype=np.float32)

    if len(msg.position) >= 6:
        arm = [float(position) for position in msg.position[:6]]
        gripper = float(msg.position[6]) if len(msg.position) >= 7 else fallback_gripper
        return np.asarray(arm + [float(gripper)], dtype=np.float32)

    return None


def extract_action_14d_from_commands(left_cmd_msg, right_cmd_msg, fallback_state_14d):
    left = extract_command_7d(
        left_cmd_msg,
        LEFT_ARM_JOINTS,
        LEFT_GRIPPER_JOINT,
        fallback_gripper=float(fallback_state_14d[6]),
    )
    right = extract_command_7d(
        right_cmd_msg,
        RIGHT_ARM_JOINTS,
        RIGHT_GRIPPER_JOINT,
        fallback_gripper=float(fallback_state_14d[13]),
    )
    if left is None or right is None:
        return None
    return np.concatenate([left, right]).astype(np.float32)


def read_ros1_bag(bag_path, image_topics, joint_state_topic, cmd_topics):
    import rosbag
    from cv_bridge import CvBridge

    bridge = CvBridge()

    image_buffers = {
        "head": [],
        "left_wrist": [],
        "right_wrist": [],
    }
    joint_states = []
    command_buffers = {
        "left": [],
        "right": [],
    }

    topics = list(image_topics.values()) + [joint_state_topic] + list(cmd_topics.values())

    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            timestamp = get_stamp_sec(msg, t)

            if topic == joint_state_topic:
                joint_states.append((timestamp, msg))
            elif topic == cmd_topics["left"]:
                command_buffers["left"].append((timestamp, msg))
            elif topic == cmd_topics["right"]:
                command_buffers["right"].append((timestamp, msg))

            elif topic in image_topics.values():
                camera_key = None
                for key, ros_topic in image_topics.items():
                    if topic == ros_topic:
                        camera_key = key
                        break

                if camera_key is None:
                    continue

                # ROS image 转 OpenCV
                # 如果你的图像本来是 rgb8，可以直接用 rgb8；
                # 如果显示颜色反了，可以改成 bgr8 后再 cvtColor。
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                image = resize_rgb(image, size=(224, 224))

                image_buffers[camera_key].append((timestamp, image))

    for key in image_buffers:
        image_buffers[key].sort(key=lambda x: x[0])

    joint_states.sort(key=lambda x: x[0])
    for key in command_buffers:
        command_buffers[key].sort(key=lambda x: x[0])

    return image_buffers, joint_states, command_buffers


def create_lerobot_dataset(repo_id, fps):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="yuanyou2",
        fps=fps,
        features={
            "observation.images.head": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_wrist": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_wrist": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": [YUANYOU2_MOTOR_NAMES],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": [YUANYOU2_MOTOR_NAMES],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    return dataset


def convert_bag_to_episode(
    dataset,
    bag_path,
    fps,
    task,
    image_topics,
    joint_state_topic,
    cmd_topics,
    action_source,
    max_command_age_sec,
):
    image_buffers, joint_states, command_buffers = read_ros1_bag(
        bag_path,
        image_topics=image_topics,
        joint_state_topic=joint_state_topic,
        cmd_topics=cmd_topics,
    )

    if not joint_states:
        raise RuntimeError(f"No {joint_state_topic} found in {bag_path}")

    start_time = joint_states[0][0]
    end_time = joint_states[-1][0]
    dt = 1.0 / fps

    current_time = start_time
    added_frames = 0
    command_action_frames = 0
    next_state_action_frames = 0

    while current_time < end_time - dt:
        # --- Step 1: collect nearest items for each sensor independently ---
        head_item = nearest_timed_item(image_buffers["head"], current_time)
        left_item = nearest_timed_item(image_buffers["left_wrist"], current_time)
        right_item = nearest_timed_item(image_buffers["right_wrist"], current_time)
        joint_item = nearest_timed_item(joint_states, current_time)
        next_joint_item = nearest_timed_item(joint_states, current_time + dt)

        if any(x is None for x in (head_item, left_item, right_item, joint_item, next_joint_item)):
            current_time += dt
            continue

        # --- Step 2: compute common reference timestamp ---
        # Use the latest actual sensor timestamp as the shared reference,
        # then re-match all sensors to this reference for temporal consistency.
        # This mirrors the get_frame() logic in interface.py.
        ref_time = max(
            head_item[0],
            left_item[0],
            right_item[0],
            joint_item[0],
        )

        # --- Step 3: re-match all sensors to the common reference time ---
        head = nearest_item(image_buffers["head"], ref_time)
        left_wrist = nearest_item(image_buffers["left_wrist"], ref_time)
        right_wrist = nearest_item(image_buffers["right_wrist"], ref_time)
        joint_msg = nearest_item(joint_states, ref_time)
        next_joint_msg = nearest_item(joint_states, ref_time + dt)

        if any(x is None for x in (head, left_wrist, right_wrist, joint_msg, next_joint_msg)):
            current_time += dt
            continue

        state_14d = extract_state_14d(joint_msg)
        action_14d = None

        if action_source in ("command", "command_or_next_state"):
            # Action commands also aligned to ref_time for consistency.
            left_cmd = nearest_item(command_buffers["left"], ref_time, max_delta=max_command_age_sec)
            right_cmd = nearest_item(command_buffers["right"], ref_time, max_delta=max_command_age_sec)
            action_14d = extract_action_14d_from_commands(left_cmd, right_cmd, state_14d)
            if action_14d is not None:
                command_action_frames += 1

        if action_14d is None:
            if action_source == "command":
                current_time += dt
                continue
            action_14d = extract_state_14d(next_joint_msg)
            next_state_action_frames += 1

        dataset.add_frame(
            {
                "observation.images.head": head,
                "observation.images.left_wrist": left_wrist,
                "observation.images.right_wrist": right_wrist,
                "observation.state": state_14d,
                "action": action_14d,
                "task": task,
            }
        )

        added_frames += 1
        current_time += dt

    dataset.save_episode()
    print(
        f"{bag_path}: saved {added_frames} frames "
        f"(command_actions={command_action_frames}, next_state_actions={next_state_action_frames})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", type=str)
    parser.add_argument("--output", "-o", type=str, default="eeason396/yuanyou2_test")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--task", type=str, default="pick a cube and place it on another cube")
    parser.add_argument(
        "--action-source",
        choices=("command_or_next_state", "command", "next_state"),
        default="command_or_next_state",
        help="Use recorded /left|right/joint_cmd actions when available, otherwise fall back to next-state actions.",
    )
    parser.add_argument("--max-command-age-sec", type=float, default=0.25)
    parser.add_argument("--head-image-topic", default=IMAGE_TOPICS["head"])
    parser.add_argument("--left-wrist-image-topic", default=IMAGE_TOPICS["left_wrist"])
    parser.add_argument("--right-wrist-image-topic", default=IMAGE_TOPICS["right_wrist"])
    parser.add_argument("--joint-state-topic", default=JOINT_STATE_TOPIC)
    parser.add_argument("--left-cmd-topic", default=CMD_TOPICS["left"])
    parser.add_argument("--right-cmd-topic", default=CMD_TOPICS["right"])
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    bag_files = sorted(bag_dir.glob("*.bag"))

    if not bag_files:
        raise RuntimeError(f"No .bag files found in {bag_dir}")

    dataset = create_lerobot_dataset(args.output, args.fps)
    image_topics = {
        "head": args.head_image_topic,
        "left_wrist": args.left_wrist_image_topic,
        "right_wrist": args.right_wrist_image_topic,
    }
    cmd_topics = {
        "left": args.left_cmd_topic,
        "right": args.right_cmd_topic,
    }

    for bag_path in bag_files:
        print(f"Converting {bag_path}")
        convert_bag_to_episode(
            dataset,
            str(bag_path),
            args.fps,
            args.task,
            image_topics=image_topics,
            joint_state_topic=args.joint_state_topic,
            cmd_topics=cmd_topics,
            action_source=args.action_source,
            max_command_age_sec=args.max_command_age_sec,
        )

    print(f"Done. Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
