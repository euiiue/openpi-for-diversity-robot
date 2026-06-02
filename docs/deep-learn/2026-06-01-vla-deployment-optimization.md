# VLA 部署工程优化：从 openpi-agilex 到 openpi_yuanyou2 的改进移植

## 任务概览

- **目标**：将 agilexrobotics/openpi-agilex 中的工程优化移植到 euiiue/openpi_yuanyou2，提升 VLA（Vision-Language-Action）模型在 Yuanyou2 双臂机器人上的部署质量，但不改动 ROS1 架构。
- **当前状态**：已完成 6 个文件的修改（+891 / -187 行），所有文件通过 Python 语法检查。
- **涉及系统**：ROS1 (rospy)、OpenPI 模型服务、LeRobot 数据集、Yuanyou2 双 Piper 机械臂 + 三路相机。
- **关键文件**：
  - [examples/yuanyou2/interface.py](examples/yuanyou2/interface.py) — 核心重写
  - [examples/yuanyou2/env.py](examples/yuanyou2/env.py) — 环境适配层
  - [examples/yuanyou2/main.py](examples/yuanyou2/main.py) — 部署入口
  - [examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py](examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py) — 数据转换
  - [examples/yuanyou2/check_ros_setup.py](examples/yuanyou2/check_ros_setup.py) — 预检查
  - [packages/openpi-client/src/openpi_client/websocket_client_policy.py](packages/openpi-client/src/openpi_client/websocket_client_policy.py) — RTC 客户端
  - [scripts/serve_policy.py](scripts/serve_policy.py) — 策略服务器
- **结论一句话**：将 agilex 的传感器同步、运动插值、图像预处理、时间对齐等工程实践统一移植到了 yuanyou2 的 ROS1 代码中，训练和推理的数据流在各环节保持了一致的时间同步策略。

---

## 已确认事实与推断

### 已确认事实

1. **openpi-agilex 使用 ROS2 (rclpy)**，openpi_yuanyou2 使用 **ROS1 (rospy)**。ROS1 已 EOL，但因为 Jetson 硬件和现有驱动栈限制，本次移植保留了 ROS1。
2. **agilex 的控制架构是自定义 ROS2 节点**，绕过 openpi_client Runtime 框架；yuanyou2 保持了 **Runtime 框架**（Environment → PolicyAgent → Runtime）。
3. **agilex 使用 BlockingDeque** 解耦 ROS 回调和推理线程，yuanyou2 原来直接用 `threading.Lock` + 简单的 dict 缓存。
4. **agilex 有独立控制线程 200Hz 插值**，yuanyou2 原来直接发布 JointState 无插值。
5. **agilex 有 RTC (Real-Time Chunking) guidance**，yuanyou2 原来没有。
6. **agilex 支持 TensorRT/ONNX 边缘推理**（针对 Jetson），yuanyou2 只支持远程 WebSocket 推理。
7. **两个仓库的训练配置结构相同**（都是 `TrainConfig` + `DataConfigFactory`），但 yuanyou2 有自己的 `LeRobotYuanyou2DataConfig` 和 14 维 state/action 布局。

### 推断/嫌疑

- **RTC guidance 的完整效果需要在服务器端策略模型支持后才能验证**。当前客户端已具备发送 RTC 数据的能力（通过 `infer_with_rtc_guidance` 封装到 observation dict），但远端策略服务器是否使用了这些字段取决于 Pi0 模型的 serving 实现。
- **TensorRT/ONNX 边缘推理**暂未移植，因为需要 CUDA/TensorRT 工具链和模型导出流程，这属于模型工程而非部署工程。

---

## 端到端链路

### 训练数据流

```text
rosbag 录制 (Jetson)
  -> convert_yuanyou2_data_to_lerobot.py (时间对齐, 转 LeRobot 格式)
  -> compute_norm_stats.py (计算归一化统计量)
  -> train.py (LoRA/Full fine-tune π₀.₅)
  -> checkpoint 保存
```

### 推理数据流（改进后）

```text
ROS1 相机话题 (3路, ~30Hz)           ROS1 /joint_states (~100Hz)
        │                                    │
        ▼                                    ▼
  BlockingDeque[head]                 BlockingDeque[joint_state]
  BlockingDeque[left_wrist]                 │
  BlockingDeque[right_wrist]                │
        │                                    │
        └──────────┬─────────────────────────┘
                   ▼
            get_frame()
    ┌─ 找到所有传感器最新时间戳的 max 作为 ref_time
    └─ 每个传感器各自匹配到 ref_time（最近邻）
                   │
                   ▼
        _process_camera_image()
    ┌─ center_crop_or_pad → 640×480
    ├─ JPEG compress → decompress (模拟训练分布)
    └─ resize_with_pad → 224×224
                   │
                   ▼
            get_observation()
    {"state": (14,), "images": {head, left_wrist, right_wrist}}
                   │
                   ▼
         WebSocket → Policy Server (GPU)
                   │
                   ▼
            动作 chunk 返回
                   │
                   ▼
          publish_action(action_14d)
    ┌─ 存入 _current_chunk
    └─ _chunk_step = 0, _target_reached = False
                   │
                   ▼
        _control_loop() @ 200Hz
    ┌─ _interpolate_step(current → target)
    │   ├─ max_delta > 0.5 rad: 线性步进, 每轴限速
    │   ├─ max_delta < 0.001 rad: 直接到位
    │   └─ 0.001~0.5 rad: 多项式外推 (k=3)
    ├─ 到达 target → _chunk_step += 1
    └─ chunk 耗尽 → 保持最后位置
                   │
                   ▼
         /left/joint_cmd  +  /right/joint_cmd
                   │
                   ▼
           Piper 机械臂执行
```

### 逐段说明

#### 1. 传感器采集与缓冲

- **输入**：3 路 RGB 相机话题 + `/joint_states`
- **输出**：`BlockingDeque` 中的 `(timestamp, data)` 元组
- **关键代码**：[interface.py:298-317](examples/yuanyou2/interface.py#L298-L317) — `_setup_subscribers()`, `_image_callback()`, `_joint_state_callback()`
- **失败模式**：若某个相机断流，`get_frame()` 返回 None，`get_observation()` 向上抛 `RuntimeError`

#### 2. 多传感器时间同步 (get_frame)

- **输入**：各 `BlockingDeque` 中积压的 sensor 消息
- **输出**：`{"images": {name: ndarray}, "joint_state": JointState, "timestamp": float}`
- **关键代码**：[interface.py:417-455](examples/yuanyou2/interface.py#L417-L455)
- **算法**：
  1. 找到所有队列最新消息时间戳的 max → `ref_time`
  2. 对每个队列做 `_get_closest_msg(queue, ref_time)` → 弹出并返回时间戳最接近 `ref_time` 的消息
  3. 这保证了所有传感器数据来自同一个真实时刻
- **失败模式**：若某个队列为空，返回 None

#### 3. 图像预处理管线

- **输入**：RGB `ndarray (H, W, 3)`
- **输出**：`ndarray (224, 224, 3)` uint8
- **关键代码**：[interface.py:461-473](examples/yuanyou2/interface.py#L461-L473)
- **管线**：
  1. `_center_crop_or_pad(image, 480, 640)` — 中心裁剪/填充到 VGA
  2. `jpeg_compress_decompress(image, quality=95)` — JPEG 压缩→解压，模拟训练数据的压缩伪影
  3. `resize_with_pad(image, 224, 224)` — 保持宽高比缩放后居中填充到 224×224
- **为什么需要 JPEG 管线**：训练数据来自 rosbag 中的 JPEG 压缩图像（USB 相机通常内部 JPEG 编码），推理时若不模拟这个压缩过程，图像分布与训练不一致，模型性能会下降。

#### 4. 动作插值执行

- **输入**：`publish_action(actions)` — 14 维目标关节角
- **输出**：200 Hz 的 `/left/joint_cmd` + `/right/joint_cmd`
- **关键代码**：
  - [interface.py:604-665](examples/yuanyou2/interface.py#L604-L665) — `_control_loop()`, `_publish_interpolated_step()`
  - [interface.py:681-743](examples/yuanyou2/interface.py#L681-L743) — `_interpolate_step()`, `_polynomial_interpolate_step()`
- **插值策略**：
  - `max_delta > 0.5 rad`（~28.6°）：大步距，每轴限速线性步进（joint_step=0.005 rad, gripper_step=0.01）
  - `max_delta < 0.001 rad`（~0.06°）：直接到位
  - 中间范围：用历史 k=3 个命令点的多项式拟合外推，70% 外推 + 30% 直接趋向目标
- **抢占式发布**：新 `publish_action()` 调用直接替换 `_current_chunk` 并将 `_chunk_step` 归零，控制线程立即转向新目标

#### 5. RTC Guidance 数据流

- **输入**：上一轮 action chunk + 已执行步数
- **输出**：在 observation dict 中附加 `rtc_guidance` 字段
- **关键代码**：
  - [interface.py:569-573](examples/yuanyou2/interface.py#L569-L573) — 存储 prev_action_chunk
  - [websocket_client_policy.py:57-82](packages/openpi-client/src/openpi_client/websocket_client_policy.py#L57-L82) — `infer_with_rtc_guidance()`
  - [env.py:90-94](examples/yuanyou2/env.py#L90-L94) — 在 observation 中附加 RTC 数据

---

## 标准工作流

### Step 1: 数据采集

- **做了什么**：在 Jetson 上启动三路相机和 TF，人工遥控操作录制 rosbag
- **为什么需要**：VLA 模型需要 demonstration 数据来 fine-tune
- **不做会怎样**：没有训练数据，只能用零样本 baseline，任务成功率极低
- **如何验证**：`check_ros_setup.py --check-rates` 确认所有 topic 正常发布

### Step 2: 数据转换（已改进时间对齐）

- **做了什么**：[convert_yuanyou2_data_to_lerobot.py](examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py) 读取 rosbag，按 10Hz 采样，做公共参考时间对齐后写入 LeRobot 格式
- **为什么需要**：rosbag 中各传感器时间戳不同步（相机 30Hz、关节 100Hz、命令不同频率），直接采样会导致同一 frame 内数据来自不同时刻
- **不做会怎样**：模型学到的是时间错位的状态-动作映射，部署时推理质量差
- **如何验证**：转换后检查数据集的 frame 数量 = bag 时长 × fps

### Step 3: 训练

- **做了什么**：修改 config.py 中的 `repo_id`，运行 `compute_norm_stats.py` 后 `train.py`
- **为什么需要**：fine-tune π₀.₅ 基座模型到 yuanyou2 特定硬件
- **不做会怎样**：策略输出不符合你的机械臂运动学特性
- **如何验证**：训练 loss 下降、checkpoint 正常保存

### Step 4: 部署推理

- **做了什么**：GPU 服务器启动 `serve_policy.py`，Jetson 启动 `main.py`
- **为什么需要**：GPU 做模型推理，Jetson 做传感器采集和运动控制
- **不做会怎样**：Jetson 显存通常不够跑 π₀.₅
- **如何验证**：Jetson 端日志显示 "Connected to policy server"，控制线程正常启动

---

## 关键代码与参数

| 项目 | 位置 | 作用 | 现值/建议值 | 风险 |
|------|------|------|-------------|------|
| `interpolation_hz` | `main.py --interpolation-hz` | 控制线程频率 | 200 Hz | 过低→运动不平滑；过高→CPU 过载 |
| `arm_steps_length` | `interface.py:147` | 每步最大关节位移 | `[0.005]*6 + [0.01]` rad | 过大→运动突变；过小→运动太慢 |
| `use_jpeg_pipeline` | `main.py --use-jpeg-pipeline` | JPEG 压缩管线开关 | True | 关闭→推理图像与训练分布不一致 |
| `jpeg_quality` | `main.py --jpeg-quality` | JPEG 压缩质量 | 95 | 太低→图像模糊；太高→压缩伪影不够 |
| `poly_k` | `interface.py:216` | 多项式插值历史点数 | 3 | 太大→过拟合历史轨迹；太小→退化为线性 |
| `preemptive_publishing` | `main.py --preemptive-publishing` | 抢占式运动 | True | 关闭→旧轨迹无法被打断，新指令延迟大 |
| `use_rtc_guidance` | `main.py --use-rtc-guidance` | RTC 时序一致性 | False（需服务端支持） | 无服务端支持时无效 |
| `action_source` | `convert_*.py --action-source` | 训练 action 来源 | `command_or_next_state` | `command` 模式：若录制时缺少 cmd topic 则无 action |
| `max_command_age_sec` | `convert_*.py --max-command-age-sec` | 命令时效窗口 | 0.25s | 过大→用过期的命令做 action；过小→有效命令被丢弃 |
| `control_frequency` | `main.py --control-frequency` | 策略查询频率 | 10 Hz | 与训练 fps 不一致→时序分布偏移 |
| `open_loop_horizon` | `main.py --open-loop-horizon` | 开环执行步数 | 10 | 大于 action_horizon → 策略被过度查询 |

---

## 知识地图

```text
VLA 部署工程
  ├── 工程链路
  │   ├── ROS1 话题通信 (Publisher/Subscriber)
  │   ├── 多线程并发 (BlockingDeque, Control Thread, Callback)
  │   ├── WebSocket 远程推理 (msgpack 序列化)
  │   └── LeRobot 数据集格式
  ├── 核心算法/控制
  │   ├── 多传感器时间同步 (Closest-Timestamp Matching)
  │   ├── 关节空间插值 (多项式 + 线性 + SLERP)
  │   ├── JPEG 压缩管线 (训练-推理分布对齐)
  │   └── RTC Guidance (时序一致性约束)
  ├── 数学/物理基础
  │   ├── 最近邻搜索 (bisect + 时间戳差值)
  │   ├── 多项式拟合与外推 (numpy.polyfit/polyval)
  │   ├── 球面线性插值 SLERP (姿态平滑)
  │   └── 速度限制与步长控制 (运动学约束)
  └── 可观测现象与调参方法
      ├── 运动抖动 → 增大 interpolation_hz 或减小 arm_steps_length
      ├── 图像异常 → 检查 JPEG 管线是否与训练一致
      └── 推理延迟 → 检查网络和 GPU 利用率
```

---

## 知识分解

### 1. 多传感器时间同步 (Multi-Sensor Temporal Synchronization)

#### Why

机器人的相机（30Hz）、关节编码器（100Hz）、命令话题（不定频率）各自以不同频率发布。如果每个传感器独立取"最新值"来构建观测，同一帧里的头部图像、腕部图像、关节角度可能来自相差数十毫秒的不同时刻。VLA 模型收到的状态与实际物理状态不一致，推理质量必然下降。

#### What

**直观**：给所有传感器拍一张"快照"，保证这张快照里的所有数据来自同一时刻。

**精确**：将所有传感器队列中最新消息的时间戳取最大值作为参考时间 `ref_time`，对每个传感器队列做最近邻搜索，找到时间戳最接近 `ref_time` 的消息，弹出该消息及其之前的所有旧消息。

#### How

在 [interface.py:417-455](examples/yuanyou2/interface.py#L417-L455) 中：

```python
def get_frame(self) -> dict | None:
    # Step 1: 找到公共参考时间
    ref_time = 0.0
    for q in self._image_queues.values():
        last = q.peek_last()
        if last is not None:
            ref_time = max(ref_time, last[0])
    last_joint = self._joint_state_queue.peek_last()
    if last_joint is not None:
        ref_time = max(ref_time, last_joint[0])

    # Step 2: 每个传感器匹配到 ref_time
    images = {}
    for cam_name, queue in self._image_queues.items():
        result = self._get_closest_msg(queue, ref_time)
        images[cam_name] = result[1]

    joint_result = self._get_closest_msg(self._joint_state_queue, ref_time)
    return {"images": images, "joint_state": joint_result[1], "timestamp": ref_time}
```

`_get_closest_msg()` ([interface.py:383-402](examples/yuanyou2/interface.py#L383-L402)) 遍历队列快照，找到时间戳差值最小的 item，弹出它及其之前的所有旧消息。

数据转换脚本 [convert_yuanyou2_data_to_lerobot.py](examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py) 使用相同的策略（两阶段：先独立采样获取各传感器时间戳 → 计算 ref_time → 重新统一匹配）。

#### 前置知识

##### 1.1 ROS 时间戳 (ROS Time vs Wall Time)

- **Why**：ROS 消息自带 header.stamp，是数据产生的时刻。不同话题的时间戳来自同一时钟源（通常是系统时钟或 sim time），可以直接比较。
- **What**：`msg.header.stamp.to_sec()` 返回浮点秒数。`rospy.Time.now()` 是当前 ROS 时间。
- **How**：在 interface.py 中，图像和关节回调都提取 `msg.header.stamp.to_sec()` 作为排序 key。如果系统时钟不同步（如 Jetson NTP 故障），时间戳会严重偏差。

##### 1.2 最近邻搜索 (Nearest-Neighbor Search in Sorted Array)

- **Why**：传感器消息按时序排列，需要在 O(log n) 时间内找到最接近目标时间戳的消息。
- **What**：`bisect_left(sorted_times, target)` 返回插入位置，比较 `items[idx-1]` 和 `items[idx]` 哪个更近。
- **How**：`convert_yuanyou2_data_to_lerobot.py` 中的 `nearest_timed_item()` 函数演示了标准实现。

---

### 2. 关节空间运动插值 (Joint-Space Motion Interpolation)

#### Why

VLA 策略输出的相邻动作之间可能存在较大的关节角度差（例如从当前位置直接跳到目标位置 0.5 rad ≈ 28.6°）。如果直接将目标角度发布给机械臂控制器，会产生突变运动（急加速→急减速），轻则抖动，重则触发驱动器过流保护或损坏减速器。

#### What

**直观**：在两个离散的目标位置之间，用高频率（200Hz）生成一系列中间位置，让机械臂平滑过渡。

**精确**：根据当前位置到目标位置的 delta 大小选择插值策略：
- 大步距 (>0.5 rad)：线性插值，每步最多移动 `arm_steps_length`（0.005 rad 关节、0.01 夹爪）
- 中等步距 (0.001~0.5 rad)：用历史 k=3 个命令点拟合多项式，外推下一步位置
- 小步距 (<0.001 rad)：直接到达目标

#### How

在 [interface.py:604-665](examples/yuanyou2/interface.py#L604-L665) 中：

```python
def _control_loop(self):
    """High-frequency loop that publishes interpolated joint commands."""
    rate = rospy.Rate(self._interpolation_hz)  # 200 Hz
    while self._ctrl_running and not rospy.is_shutdown():
        self._publish_interpolated_step()
        rate.sleep()

def _publish_interpolated_step(self):
    # 读取当前目标
    target = chunk[step]
    # 读取当前插值位置
    current_left, current_right = self._get_current_joint_positions()
    # 计算一步插值
    left_cmd = self._interpolate_step(current_left, target_left, history_left)
    right_cmd = self._interpolate_step(current_right, target_right, history_right)
    # 发布
    self._publish_joint_cmd(left_cmd, right_cmd)
    # 到达目标后 advance 到 chunk 的下一步
    if dist < tolerance:
        self._chunk_step += 1
```

`_interpolate_step()` ([interface.py:681-710](examples/yuanyou2/interface.py#L681-L710)) 实现了三种模式的分支逻辑，`_polynomial_interpolate_step()` ([interface.py:712-743](examples/yuanyou2/interface.py#L712-L743)) 负责多项式外推，并将外推结果与直接趋向目标做 70/30 混合，保证收敛性。

#### 前置知识

##### 2.1 多项式插值与外推 (Polynomial Interpolation and Extrapolation)

- **Why**：纯线性插值会产生恒定速度的运动，在起点和终点处加速度不连续（突跳），外观上表现为"机械感"。多项式插值利用历史轨迹的曲率信息产生更自然的运动。
- **What**：给定 k 个历史点 `(x_i, y_i)`，拟合一个 k-1 次多项式 `P(x)` 使得 `P(x_i) = y_i`。在 `x = k` 处评估 `P(k)` 即为对下一步位置的外推。
- **How**：`numpy.polyfit(x, y, deg=k-1)` 做最小二乘拟合，`numpy.polyval(coeffs, x_next)` 做外推。外推结果与直接趋向目标做加权平均，防止发散。

##### 2.2 速度限制与步长控制 (Velocity Limiting via Step Size Clamping)

- **Why**：机械臂每个关节有最大速度和加速度限制。如果一步之内关节角跨越太大，驱动器可能无法跟踪，导致轨迹偏差或过流报警。
- **What**：`arm_steps_length = [0.005]*6 + [0.01]` 定义了每个关节（前 6 个 arm joint + 第 7 个 gripper）在单步（1/200 秒）内允许的最大位移。关节 0.005 rad/step × 200 Hz = 1.0 rad/s ≈ 57°/s 最大速度。
- **How**：在 `_interpolate_step` 中，对大步距计算 `scale = min(step_limits[i] / delta[i])`，将位移缩放到每个轴都不超过限制。

---

### 3. JPEG 压缩管线与训练-推理分布对齐 (JPEG Pipeline for Train-Inference Distribution Matching)

#### Why

USB 相机（如 Orbbec DCW、Intel RealSense D435）内部使用硬件 JPEG 编码器压缩图像后通过 USB 传输。rosbag 中存储的图像已经经过了 JPEG 压缩→解压的过程（有 compression artifacts）。但推理时从 ROS topic 直接订阅的图像是 cv_bridge 解码后的"干净"图像。如果不模拟 JPEG 压缩过程，推理图像的分布与训练数据不一致，模型会产生系统性偏差。

#### What

**直观**：在推理时对图像做一次"先压缩再解压"，人为引入与训练数据相同的 JPEG 压缩伪影。

**精确**：`cv2.imencode('.jpg', image, [quality])` 将 RGB 图像编码为 JPEG 字节流 → `cv2.imdecode(bytes)` 解码回 RGB 数组。这个过程在 224×224 尺度上引入约 0.5-2 像素值的微小变化，恰好匹配训练数据的噪声特征。

#### How

在 [interface.py:118-123](examples/yuanyou2/interface.py#L118-L123) 中：

```python
def jpeg_compress_decompress(image: np.ndarray, quality: int = 95) -> np.ndarray:
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
```

该函数在 `_process_camera_image()` 中位于 `_center_crop_or_pad(image, 480, 640)` 之后、`resize_with_pad(image, 224, 224)` 之前。

#### 前置知识

##### 3.1 JPEG 压缩原理 (Discrete Cosine Transform + Quantization)

- **Why**：JPEG 把图像分成 8×8 块，每块做 DCT 变换到频域，然后用量化表丢掉高频细节（人眼不敏感），最后熵编码。quality=95 时量化步长较小，artifact 轻微但确实存在。
- **What**：JPEG artifact 主要表现为 block boundary（8×8 块边缘）和 ringing（高对比度边缘周围的振荡）。
- **How**：`cv2.IMWRITE_JPEG_QUALITY` 控制量化表缩放，95 表示高质量（轻微 artifact），50 表示低质量（明显 artifact）。

##### 3.2 图像预处理管线顺序

- **Why**：不同分辨率操作的先后顺序影响最终图像质量。先 crop 到 VGA 再做 JPEG 压缩，模拟的是相机内部在原始分辨率下压缩的场景；如果先 resize 到 224 再 JPEG，artifact 模式会不同。
- **What**：正确的顺序是：`crop/pad → JPEG → resize`，匹配 USB 相机的物理管线。
- **How**：在 `_process_camera_image()` 中严格遵守此顺序。

---

### 4. RTC Guidance — 时序一致性约束 (Real-Time Chunking Guidance)

#### Why

VLA 模型输出一个 action chunk（如 30 步的动作序列），部署时通常只执行前 N 步（open_loop_horizon=10），然后重新查询策略。如果不告诉模型"上次的 chunk 已经执行了哪些步骤"，新产生的 chunk 可能与上一段之间出现跳变（例如第 10 步和第 11 步之间关节角差 0.3 rad）。

#### What

**直观**：每次推理时告诉模型"我刚才执行到了哪里"，让模型在规划新 chunk 时考虑与上一段轨迹的连续性。

**精确**：将 `prev_action_chunk`（上一次完整的 action chunk）和 `executed_steps`（已执行步数）作为额外输入传给策略服务器。策略内部的 attention mask 会对已执行步施加 guidance weight，使新 chunk 的起始部分与旧 chunk 平滑衔接。

#### How

在 yuanyou2 的实现中：

1. [interface.py:569-573](examples/yuanyou2/interface.py#L569-L573)：`publish_action()` 存储上一个 chunk 和已执行步数
2. [env.py:90-94](examples/yuanyou2/env.py#L90-L94)：`get_observation()` 将 RTC 数据附加到 observation dict
3. [websocket_client_policy.py:57-82](packages/openpi-client/src/openpi_client/websocket_client_policy.py#L57-L82)：`infer_with_rtc_guidance()` 封装 RTC 参数

**注意**：完整效果需要策略服务器端的 Pi0 serving 代码支持 RTC。当前实现确保客户端已具备发送能力，若服务器不支持，extra fields 被静默忽略。

---

## 调试决策树

```text
机械臂运动抖动的可能原因:

├── 关节角突变?
│   ├── 看 _interpolate_step 日志中的 max_delta
│   ├── max_delta > 0.5 rad 且频繁出现 → arm_steps_length 太大 → 减小到 0.002
│   └── max_delta < 0.001 rad 但仍有抖动 → 控制线程频率不够 → 增大 interpolation_hz
│
├── 控制线程不工作?
│   ├── 看日志是否有 "Control thread started at XXX Hz"
│   ├── 无日志 → _start_control_thread 未调用 → 检查 Yuanyou2ROS1Interface.__init__
│   └── 有日志但无运动 → 检查 _ctrl_running 是否被设为 False
│
├── 传感器时间不同步?
│   ├── 看 get_frame 返回的 timestamp 是否连续递增
│   ├── 时间戳跳跃 → 某个相机断流或 NTP 不同步
│   └── 修复：用 check_ros_setup.py --check-rates 确认话题频率
│
├── 策略输出质量差?
│   ├── 看 action chunk 的数值范围是否合理（关节角 ±π，夹爪 0~0.035）
│   ├── 数值异常 → 训练数据 norm stats 有问题
│   └── 数值正常但效果差 → 检查训练时的 action_source 是否与推理一致
│
└── JPEG 管线导致图像模糊?
    ├── 临时关闭 --no-use-jpeg-pipeline 对比
    ├── 画面改善 → jpeg_quality 太低 → 调整到 98
    └── 画面无变化 → JPEG 不是问题根因
```

---

## 常见误区

1. **误区**："interpolation_hz 越高越好"
   - **为什么容易错**：直觉上频率越高运动越平滑
   - **正确理解**：200 Hz 已经远超机械臂伺服周期（通常 1-5ms）。超过 500 Hz 时 `rospy.Rate.sleep()` 精度不足（Linux 调度器 tick 通常 250Hz-1000Hz），实际频率可能抖动。200-300 Hz 是最佳区间。

2. **误区**："JPEG 管线可有可无，质量差异肉眼不可见"
   - **为什么容易错**：95% 质量的 JPEG 在人眼看来与原始图像几乎无差别（PSNR ~45dB）
   - **正确理解**：神经网络对像素级高频 pattern 敏感。没有 JPEG 管线的推理图像比训练图像"干净"，模型可能过度依赖这些在真实场景中不存在的细节。关闭 JPEG 管线后需要重新评估模型性能。

3. **误区**："数据转换时 action_source 用 next_state 就够了"
   - **为什么容易错**：next_state 是执行后的关节位置，大致等于目标位置
   - **正确理解**：如果录制时机械臂在运动过程中，next_state 是插值过程中的中间状态，不等于遥操作的命令目标。使用 `command_or_next_state` 优先取实际命令，只在命令缺失时退回 next_state。

---

## 思考与问答

1. 【基础】`get_frame()` 中 `ref_time` 为什么取所有传感器最新时间戳的**最大值**而不是最小值或平均值？
2. 【基础】`_interpolate_step()` 中为什么对 max_delta > 0.5 rad 使用线性插值，而不是对所有情况都使用多项式插值？
3. 【进阶】如果训练时 fps=10 但推理时 `control_frequency=20`，会有什么问题？代码中哪些参数需要同步调整？
4. 【进阶】`_polynomial_interpolate_step()` 中为什么将外推结果与直接趋向目标做 70/30 混合，而不是 100% 外推？
5. 【综合】假设你在 Jetson 上看到机械臂运动卡顿，`rostopic hz /left/joint_cmd` 显示只有 50Hz 而不是预期的 200Hz。请列出排查步骤。
6. 【综合】如果把 yuanyou2 从 ROS1 迁移到 ROS2，当前 interface.py 中哪些组件需要完全重写，哪些可以复用？

---

## 术语表

| 中文 | English | 一句话解释 |
|------|---------|-----------|
| 视觉-语言-动作模型 | VLA (Vision-Language-Action) | 输入图像+文本指令，输出机器人动作的端到端模型 |
| 动作块 | Action Chunk | 模型一次输出多步动作（如 30 步），而非单步 |
| 开环执行 | Open-Loop Execution | 执行动作块时不重新观测，执行 N 步后再查询策略 |
| 时间同步 | Temporal Synchronization | 多传感器数据对齐到同一时刻 |
| 关节空间插值 | Joint-Space Interpolation | 在两个目标关节角之间生成平滑的中间位置 |
| 抢占式发布 | Preemptive Publishing | 新指令到达时立即取消正在执行的旧轨迹 |
| 实时分块引导 | RTC Guidance | 将上一轮 action chunk 的执行进度反馈给策略，保证时序连续性 |
| 训练-推理分布偏移 | Train-Inference Distribution Shift | 训练数据和推理数据在统计特征上的系统性差异 |
| 归一化统计量 | Norm Stats | 训练数据的均值和标准差，用于对输入做 z-score 归一化 |
| 最近邻搜索 | Nearest-Neighbor Search | 在有序列表中查找最接近目标值的元素 |
| 多项式外推 | Polynomial Extrapolation | 用历史数据点拟合的多项式预测未来值 |
| JPEG 压缩伪影 | JPEG Compression Artifact | 有损压缩引入的图像块效应和振铃噪声 |
| LeRobot | LeRobot | HuggingFace 的机器人数据集格式和工具库 |
| π₀.₅ (Pi05) | Pi0.5 | Physical Intelligence 的 VLA 基座模型 |
| LoRA | Low-Rank Adaptation | 只训练低秩矩阵的低成本微调方法 |
