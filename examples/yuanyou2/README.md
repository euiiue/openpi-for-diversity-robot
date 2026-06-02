# Yuanyou2 OpenPI playlist
<a href="https://zread.ai/euiiue/openpi_yuanyou2" target="_blank"><img src="https://img.shields.io/badge/Ask_Zread-_.svg?style=plastic&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff" alt="zread"/></a>

本目录包含 Yuanyou2 双 Piper 轮臂机器人接入 OpenPI 的 ROS1 运行客户端和 rosbag 转 LeRobot 数据集工具。

## 运行约定

当前策略输入状态和输出动作都是 14 维：

```text
[left_joint1..left_joint6, left_gripper,
 right_joint1..right_joint6, right_gripper]
```

左右夹爪标量都使用各自的 `joint7`。URDF 中的 `joint8` 是联动的另一侧手指，本 OpenPI 接口不单独控制 `joint8`。

三路图像输入约定为：

```text
observation/images/head
observation/images/left_wrist
observation/images/right_wrist
```

头部相机固定在机器人 TF 树上，不再走手眼标定流程；双腕相机使用已经保存并验证过的 wrist static TF。

## 采集前启动

在 Jetson 上先加载 ROS 环境：

```bash
source /opt/ros/noetic/setup.bash
source ~/yuanyou2_ws/devel/setup.bash
```

启动三路相机和双腕 TF：

```bash
roslaunch yuanyou2 three_cameras.launch
roslaunch yuanyou2 camera_tf.launch publish_head:=false
```

采集前运行预检查脚本：

```bash
python3 examples/yuanyou2/check_ros_setup.py
```

如果脚本提示相机 topic、`camera_info`、`/joint_states` 或系统时间异常，需要先修好再采集正式数据。Jetson 时间如果还停在 1970，会影响 bag 文件时间戳、日志排查和后续数据整理。

## 建议录制的 Bag Topic

至少录制下面这些 topic：

```bash
rosbag record -O yuanyou2_demo.bag \
  /joint_states \
  /left/joint_cmd \
  /right/joint_cmd \
  /head_camera/usb_cam/image_raw \
  /head_camera/usb_cam/camera_info \
  /left_wrist_d435/color/image_raw \
  /left_wrist_d435/color/camera_info \
  /right_wrist_d435/color/image_raw \
  /right_wrist_d435/color/camera_info \
  /tf \
  /tf_static
```

转换脚本会优先把 `/left/joint_cmd` 和 `/right/joint_cmd` 作为真实 action。若这两个 command topic 缺失，默认会退回到下一帧 `/joint_states` 近似 action。

## 转换 Bag

转换脚本依赖 ROS1 的 `rosbag` 和 `cv_bridge`。在 Jetson 上转换时通常需要先 source ROS Noetic；如果在训练机上转换，需要使用能 import ROS1 Python 包的虚拟环境或容器。

```bash
source /opt/ros/noetic/setup.bash
uv run examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py /path/to/bag_dir \
  --output eeason396/yuanyou2_test \
  --fps 10 \
  --task "pick a cube and place it on another cube"
```

如果希望严格只使用实际 command action，使用：

```bash
uv run examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py /path/to/bag_dir \
  --action-source command
```

如果你的实际 topic 名称和默认值不同，可以通过参数覆盖：

```bash
uv run examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py /path/to/bag_dir \
  --head-image-topic /your/head/image_raw \
  --left-wrist-image-topic /your/left_wrist/image_raw \
  --right-wrist-image-topic /your/right_wrist/image_raw \
  --joint-state-topic /joint_states \
  --left-cmd-topic /left/joint_cmd \
  --right-cmd-topic /right/joint_cmd
```

## 训练

转换完成后，先计算归一化统计量，再开始训练：

```bash
uv run scripts/compute_norm_stats.py pi05_yuanyou2_lora_finetune
uv run scripts/train.py pi05_yuanyou2_lora_finetune --exp_name first_run
```

开始训练前，需要把 Yuanyou2 训练配置里的 `repo_id="your_username/yuanyou2_dataset"` 改成你真实的 LeRobot 数据集名称。

## 部署运行

在策略服务器机器上启动 policy server：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_yuanyou2_lora_finetune \
  --policy.dir=checkpoints/pi05_yuanyou2_lora_finetune/first_run/20000
```

在 Jetson 上启动机器人侧客户端：

```bash
python3 examples/yuanyou2/main.py \
  --remote-host <policy-server-ip> \
  --prompt "pick a cube and place it on another cube"
```

如果运行时 topic 名称和默认配置不一致，也可以在客户端启动时覆盖：

```bash
python3 examples/yuanyou2/main.py \
  --remote-host <policy-server-ip> \
  --head-image-topic /your/head/image_raw \
  --left-wrist-image-topic /your/left_wrist/image_raw \
  --right-wrist-image-topic /your/right_wrist/image_raw \
  --prompt "pick a cube and place it on another cube"
```
