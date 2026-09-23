# OpenPI CR3/O6：数据处理、训练与实机部署操作手册

> 适用仓库：[euiiue/openpi-for-diversity-robot](https://github.com/euiiue/openpi-for-diversity-robot)，分支 `cr5-o6-pi05`。本文按该分支的当前脚本与配置编写。实际部署前，以所使用提交的代码为准。

## 一、项目概述与执行流程

本项目基于 OpenPI π0.5，构建 CR3/CR3A 六轴机械臂与 LinkerHand O6 灵巧手的 VLA 训练和部署基线。遥操作及数据采集主要由 [gello_CR](https://github.com/euiiue/gello_CR) 完成；本仓库负责将已完成的 LeRobot v3 数据转换为训练兼容格式、检查数据、计算归一化统计量、微调 π0.5，并通过 WebSocket 提供推理服务和实机客户端。

```text
gello_CR 遥操作、采集 LeRobot v3 数据（20 Hz）
    ↓
核查成功示教、关节顺序、任务文本、相机数据
    ↓
convert_data_to_lerobot.py：v3 → 本项目兼容的 v2.1 数据集
    ↓
check_dataset.py：检查字段、数值与训练变换
    ↓
compute_norm_stats.py：计算归一化统计量
    ↓
train.py：π0.5 LoRA 训练、保存 Checkpoint
    ↓
serve_policy.py：GPU 服务器启动 WebSocket 策略服务
    ↓
main_rtc.py：机器人端预览 → 人工审核 → 实机执行
```

### 1. 先区分两台机器

| 位置 | 使用环境 | 执行内容 |
|---|---|---|
| GPU 训练／推理服务器 | Linux、NVIDIA GPU、Python 3.11 + `uv` | 转换数据、检查数据、计算统计量、训练、启动 WebSocket 服务 |
| 机器人控制电脑 | Linux x86_64、Python 3.12 虚拟环境 | 连接 NRC 控制柜、O6、三路 RealSense、WebSocket 服务，预览与运行 |

两者可以是同一台电脑。若分开部署，两台机器都需要有本项目代码，但**机器人端不需要运行 `uv sync` 或安装完整 JAX 训练环境**。机器人端的 NRC 厂商扩展依赖 CPython 3.12，不应直接用训练环境启动。

### 2. 哪些名称与地址需要修改

| 参数 | 示例 | 含义／修改时机 |
|---|---|---|
| `CR3_O6_SOURCE_ROOT_1` | `data/raw/session_001` | **每批数据**：真实的 gello_CR 原始 v3 会话根目录 |
| `CR3_O6_DATASET_ID` | `local/cr3_o6_motor_20260923` | **每批转换**：给新生成的训练数据集命名；与原始目录名不必相同 |
| `repo_id` | `local/cr3_o6_motor_20260923` | **更换训练数据集时**：必须与上述数据集 ID 一致 |
| `CONFIG_NAME` | `pi05_cr3_o6_joint_abs_lora` | 训练配置名称；复用当前配置一般不修改 |
| `EXP_NAME` | `cr3_o6_motor_v1` | **每次新实验**：建议设置新的实验名，避免混淆 Checkpoint |
| `CHECKPOINT_STEP` | `45000` | **每次部署**：选择已经完整保存的步数目录 |
| `--host` | GPU 服务器的内网 IP | **跨机器部署时**：填策略服务所在电脑的真实 IP；同机才填 `127.0.0.1` |
| `CR3_O6_TASK_PROMPT` | 采集时真实任务文本 | **切换任务时**：与所选训练数据集／Checkpoint 的任务一致 |
| `robot.local.yaml` | 控制柜 IP、串口、工作空间等 | 首次部署或更换设备、作业区域时修改 |
| `camera.local.yaml` | 三个相机序列号 | 首次部署或更换相机时修改 |

---

## 二、GPU 服务器：项目初始化

### 步骤 1：下载指定分支

在 GPU 服务器的终端运行：

```bash
git clone --branch cr5-o6-pi05 \
  https://github.com/euiiue/openpi-for-diversity-robot.git openpi_cr3_o6
cd openpi_cr3_o6
```
### 步骤 2：安装并确认训练环境

提前准备 `uv`、可用的 Python 3.11，以及支持当前 JAX CUDA 12 依赖的 NVIDIA 驱动。
Python 为 3.11；JAX 可以识别预期的 GPU。如果只显示 `CpuDevice`，先处理驱动／CUDA／JAX 环境，不要开始正式训练。

### 步骤 3：加载项目路径

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"
printf 'ROOT=%s\nDATA=%s\nDATASETS=%s\nASSETS=%s\nCKPT=%s\n' \
  "$OPENPI_ROOT" "$OPENPI_DATA_HOME" "$HF_LEROBOT_HOME" \
  "$OPENPI_ASSETS_DIR" "$OPENPI_CHECKPOINT_DIR"
```

这些变量只对**当前终端**及其子进程生效。每次新开终端，先进入仓库并重新 `source examples/cr3_o6/env.sh`。默认位置为仓库内的 `data/`、`data/lerobot/`、`assets/`、`checkpoints/` 和 `.cache/jax/`。若数据存储在其他硬盘，应先设置所需路径变量，再执行 `env.sh`；例如：

```bash
export OPENPI_DATA_HOME=/your/data/disk/openpi_data   # 改成真实目录
source examples/cr3_o6/env.sh
```

此处 `/your/data/disk/openpi_data` 只是示例，不应直接照抄。

---

## 三、GPU 服务器：LeRobot v3 数据转换

### 步骤 4：定位 gello_CR 原始数据

核查：

```bash
export CR3_O6_SOURCE_ROOT_1="$OPENPI_DATA_HOME/raw/session_001"  # 必须修改为实际路径
ls "$CR3_O6_SOURCE_ROOT_1/meta/info.json"
cat "$CR3_O6_SOURCE_ROOT_1/meta/info.json"
ls "$CR3_O6_SOURCE_ROOT_1/meta/collection/" | head
```

当前转换脚本要求原始数据为**已经完成写入的 LeRobot v3.0、20 Hz、至少一个 Episode**。

### 步骤 5：设置输出数据集名称

```bash
export CR3_O6_DATASET_ID='local/cr3_o6_motor_20260923'   # 改成这批转换数据的唯一名称
export CR3_O6_DATASET_ROOT="$HF_LEROBOT_HOME/$CR3_O6_DATASET_ID"

**命名关系**：`CR3_O6_SOURCE_ROOT_1` 是采集产生的文件夹；`CR3_O6_DATASET_ID` 是本次**转换后的训练数据集 ID**，并非必须照抄采集文件夹名。该 ID 必须与随后训练配置的 `repo_id` 一致。重复转换应新建 ID，或在备份后人工处理已有目录，转换脚本不会自动覆盖。

### 步骤 6：转换与结果检查

若使用新数据集 ID，建议在执行 `check_dataset.py` 之前先按步骤 7 同步训练配置里的 `repo_id`，使校验读取到的训练变换与后续训练一致。

```bash
uv run examples/cr3_o6/convert_data_to_lerobot.py \
  --source-roots "$CR3_O6_SOURCE_ROOT_1" \
  --repo-id "$CR3_O6_DATASET_ID" \
  --output-root "$CR3_O6_DATASET_ROOT"

uv run examples/cr3_o6/check_dataset.py \
  --root "$CR3_O6_DATASET_ROOT" \
  --repo-id "$CR3_O6_DATASET_ID"
```

多段数据时，额外设置 `CR3_O6_SOURCE_ROOT_2`，并把两个路径都放到 `--source-roots` 后面。转换过程会根据 Episode 元数据、帧时间戳及视频时间戳核查输入。转换成功时打印输出目录，并在数据集内生成 `conversion.json`，记录来源、纳入／排除的 Episode 和帧数。数据检查通过后，会打印 `frames`、`episodes`、`tasks`、`model_state_shape`（`[32]`）及 `model_action_shape`（`[20,32]`）。这只能证明数据结构及变换可用，**不代表示教动作质量或实机成功率已经验证**。



## 四、GPU 服务器：配置、归一化及训练

### 步骤 7：同步训练配置的 `repo_id`

打开 `src/openpi/training/config.py`，找到 `name="pi05_cr3_o6_joint_abs_lora"` 对应的 `TrainConfig`，将该配置中的：

```python
data=LeRobotDobotCR5O6DataConfig(
    repo_id="local/cr3_o6_motor_20260923",  # 与 CR3_O6_DATASET_ID 完全一致
    joint_space=True,
    base_config=DataConfig(prompt_from_task=True),
),
```

设置为本批数据使用的 ID。**这里只是示意需要修改的字段，不要将示例代码整体覆盖原有配置。** 修改前可用 `grep -n -A 35 'name="pi05_cr3_o6_joint_abs_lora"' src/openpi/training/config.py` 定位。

### 步骤 8：计算归一化统计量

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"

export CONFIG_NAME=pi05_cr3_o6_joint_abs_lora
export EXP_NAME=cr3_o6_motor_v1        # 每次新实验改这里
export CR3_O6_DATASET_ID=local/cr3_o6_motor_20260923  # 与已转换数据集及 config.py 的 repo_id 一致

uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
ls "$OPENPI_ASSETS_DIR/$CONFIG_NAME/$CR3_O6_DATASET_ID/"
```

统计量由脚本保存在 `$OPENPI_ASSETS_DIR/$CONFIG_NAME/$CR3_O6_DATASET_ID/` 对应目录。若找不到，请先确认 `repo_id` 是否一致、是否重新计算成功。**更换数据集后需重新计算**；不要直接沿用旧数据集的统计量。该脚本会读取训练配置中的数据字段和变换规则。

### 步骤 9：启动训练

```bash

uv run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" \
```
检查 GPU 显存和利用率（另一个终端运行 `nvidia-smi`）。训练输出默认在：

```text
checkpoints/
└── pi05_cr3_o6_joint_abs_lora/
    └── cr3_o6_motor_v1/
        ├── 5000/
        ├── 10000/
        ├── ...
        └── 45000/      # 仅在此步确已保存时存在
```

实际文件夹以程序运行结果为准。终止前尽可能等到最近一个 Checkpoint 保存完成；不要仅凭出现“开始保存”字样判断写入已完成。

### 步骤 10：意外中断后续训

新开终端时重新 `source env.sh`，并**重新设置原来的** `CONFIG_NAME`、`EXP_NAME`：

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"
export CONFIG_NAME=pi05_cr3_o6_joint_abs_lora
export EXP_NAME=cr3_o6_motor_v1

uv run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" --resume
```

续训应保持训练配置、数据集映射和实验目录一致，且原 Checkpoint 必须完整。不要在同一实验目录里悄悄换数据集、动作语义或模型结构；需要更改时应另起实验，并根据需要重新计算统计量。

---

## 五、GPU 服务器：启动 Checkpoint 推理服务


### 步骤 11：开启 WebSocket 服务

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config "$CONFIG_NAME" \
  --policy.dir "$POLICY_DIR"
```

这个终端需要保持运行。当前服务监听 `0.0.0.0:8000`；`8000` 是策略服务端口。GPU 服务器通过 `ip -br addr` 查看自己实际可被机器人电脑访问的地址。
机器人电脑用 `nc -vz <GPU服务器IP> 8000` 检查 TCP 连通性，命令中的 `<GPU服务器IP>` 必须替换成实际 IP（同机连接不需要跨设备检查）。
跨网络部署还需正确配置防火墙或专用网络，不应将无认证的推理端口暴露到公网。

---

## 六、机器人控制电脑：独立安装部署环境

> **以下命令在机器人控制电脑执行，不是在远程 GPU 终端执行。** 如果使用同一台电脑，仍应保留独立的 Python 3.12 虚拟环境。

### 步骤 12：获取代码，初始化 Python 3.12（新电脑上进行配置）

```bash
git clone --branch cr5-o6-pi05 \
  https://github.com/euiiue/openpi-for-diversity-robot.git openpi_cr3_o6
cd openpi_cr3_o6
source examples/cr3_o6/env.sh

python3.12 --version
python3.12 -m venv .venv-cr3-deploy
export CR3_O6_DEPLOY_PYTHON="$OPENPI_ROOT/.venv-cr3-deploy/bin/python"
"$CR3_O6_DEPLOY_PYTHON" -m pip install --upgrade pip
"$CR3_O6_DEPLOY_PYTHON" -m pip install -r examples/cr3_o6/deploy/requirements.txt
"$CR3_O6_DEPLOY_PYTHON" -m pip install -e packages/openpi-client
"$CR3_O6_DEPLOY_PYTHON" --version
```

若已克隆，进入已有仓库即可，**不要在已有目录重复执行 `git clone`**。此步骤通常只需首次安装或更换环境时执行，**并不是每次预览都要重装依赖**。新终端重新 `source env.sh` 后，再执行一次下面的 `export` 指向已有虚拟环境：

```bash
export CR3_O6_DEPLOY_PYTHON="$OPENPI_ROOT/.venv-cr3-deploy/bin/python"
```

### 步骤 13：配置机器人、O6 与相机（新电脑上配置）

首次运行时从模板复制本地配置；已有人工审核的配置不要随意覆盖：

```bash
cp -n "$CR3_O6_CONFIG_DIR/robot.example.yaml" "$CR3_O6_CONFIG_DIR/robot.local.yaml"
cp -n "$CR3_O6_CONFIG_DIR/camera.example.yaml" "$CR3_O6_CONFIG_DIR/camera.local.yaml"
nano "$CR3_O6_CONFIG_DIR/robot.local.yaml"
nano "$CR3_O6_CONFIG_DIR/camera.local.yaml"
```

`robot.local.yaml` 至少逐项核对：

- `controller_ip`：真实 NRC 控制柜 IP，不是 GPU 服务器 IP；当前默认命令／ServoJ 端口分别为 `6001`、`7000`，实际值以控制柜为准。
- `o6_port`：真实串口设备，优先使用 `/dev/serial/by-id/...` 的稳定路径；检查 `o6_baudrate`、`o6_hand_id` 是否与设备一致。可通过 `ls -l /dev/serial/by-id/` 核查串口。
- `action_contract`：保持 `cr3_q1_q6_absolute_plus_o6_target_6d`，与所训练的 Checkpoint 匹配。
- `workspace_min_m`、`workspace_max_m`：依据实测机器人基坐标系的安全工作区域填写三轴边界，**单位为米**，不能保持 `null` 后直接启动运动。
- `max_joint_step_rad`、`max_tracking_error_rad`：必须填写根据机械臂及现场实验审核的正数，**单位为弧度**。还需核对 `joint_target_max_speed_rad_s` 和 ServoJ 运动参数。
- `o6_speed`、`o6_torque`：若设置，须同时提供六个 0～255 的整数；未审核前不要随意采用高速度或高力矩。

`camera.local.yaml` 中，按相机物理安装位置填写：`global_camera_serial`、`wrist_camera_serial`、`right_wrist_camera_serial`。使用 `rs-enumerate-devices -s` 或 RealSense Viewer 查到的真实序列号。默认采集参数为 640×480、30 FPS；程序会形成三路模型输入并检查画面时效与帧间时间差。相机序列号缺失不能正常完成三路推理预览。
---

## 七、机器人控制电脑：先预览，后执行

### 步骤 14：设置 GPU 服务器地址与任务文本（新电脑上配置）

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"
export CR3_O6_DEPLOY_PYTHON="$OPENPI_ROOT/.venv-cr3-deploy/bin/python"

export POLICY_SERVER_IP='192.168.2.100'   # 修改：实际 GPU 服务器 IP；同机用 127.0.0.1
export CR3_O6_TASK_PROMPT='填入采集时该任务的真实英文文本'  # 修改：不能直接用示例占位文本
```

任务文本不是数据集目录名。应通过转换检查输出中的 `tasks`，确认本次 Checkpoint 对应的任务描述，并填写相应原文。当前代码仅在服务器元数据包含明确 `task_prompt` 时强制比较文本；即便没有强制校验，任务不匹配也可能使推理偏离训练分布。

### 步骤 15：启动仅预览模式（不下发运动命令）

```bash
"$CR3_O6_DEPLOY_PYTHON" examples/cr3_o6/deploy/main_rtc.py \
  --host "$POLICY_SERVER_IP" --port 8000 \
  --config-dir "$CR3_O6_CONFIG_DIR" \
  --prompt "$CR3_O6_TASK_PROMPT" \
  --preview-port 8080 --preview-only
```

在**机器人控制电脑本机**打开 `http://127.0.0.1:8080`，检查三路画面的相机映射、视野、目标是否可见及是否卡顿，同时观察终端中模型推理是否持续完成。浏览器在其他电脑时，预览接口默认只绑定机器人电脑的环回地址，需通过 SSH 端口转发等方式访问。

**重要区别**：`--preview-only` 会连接 NRC、O6、三路相机与策略服务，并读取设备状态、执行模型推理；它只是不发送机器人运动命令，**不是完全脱离硬件的离线模式**。如果只想检查 Python 依赖和代码结构，请运行后面的自动化测试。预览正常后，按 `Ctrl+C` 退出。

### 步骤 16：审核后进行短时实机执行

在控制柜、O6、相机、任务文本、工作空间边界、关节运动限制以及急停可用性全部现场确认后，确保 CR3 已进入可运行的伺服状态，并执行：

```bash
"$CR3_O6_DEPLOY_PYTHON" examples/cr3_o6/deploy/main_rtc.py \
  --host "$POLICY_SERVER_IP" --port 8000 \
  --config-dir "$CR3_O6_CONFIG_DIR" \
  --prompt "$CR3_O6_TASK_PROMPT" \
  --preview-port 8080 \
  --confirm-motion
```

---


| 现象 | 首先核查 |
|---|---|
| 原始数据转换失败 | 原始路径是否指向 `meta/info.json` 所在根目录；是否 v3.0／20 Hz；Episode 是否成功、joint 模式且已审核；输出目录是否已存在 |
| `check_dataset.py` 的字段或维度错误 | 12 维关节／O6 顺序、弧度与 0～255 单位、三路图像和 `repo_id` |
| 训练找不到数据或统计量 | 当前配置的 `repo_id` 是否和转换 ID 完全一致，`HF_LEROBOT_HOME` 是否正确 |
| `--resume` 失败 | 是否使用原 `CONFIG_NAME`、`EXP_NAME`，最近一个 Checkpoint 是否完整 |
| 机器人连不上策略服务 | `--host` 是否指向 GPU 服务器而不是机器人控制柜；8000 端口、防火墙、两机网络是否通畅 |
| NRC SDK 无法导入 | Python 是否为 3.12、机器是否为 Linux x86_64、厂商 `.so`／Python 包及其动态依赖是否齐全 |
| 预览无法运行 | 三路相机序列号、控制柜 6001／7000、O6 串口和返回状态是否正常；预览也需要硬件连接 |
| `RTC` 动作过期、画面卡顿 | 检查相机帧龄、三路帧间时间差、网络与推理耗时；对照 20 Hz／20 步动作契约 |
| `start_control` 被拒绝 | 机器人是否已处于 Servo 状态 3，工作空间／关节步长／跟踪误差限值是否全部通过现场配置 |
