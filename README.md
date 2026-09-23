
# OpenPI CR3/O6

本项目基于 OpenPI 构建 CR3 机械臂与 O6 灵巧手的 VLA 模型训练及实机部署基线，后续计划按照相同流程适配 CR5、珞石等不同品牌的机械臂。

项目覆盖从机器人遥操作、数据采集、数据转换、模型训练到实机部署的完整流程。

## 一、机器人遥操作与数据采集

> 本阶段主要由 [gello_CR](https://github.com/euiiue/gello_CR) 项目完成。

### 1. 机器人遥操作

目前主要支持以下遥操作方案：

- **GELLO 主臂遥操作**
  - 六轴主臂
  - 七轴主臂
- **Inexbot 微雪主臂遥操作**
  - 四自由度主臂
- **Inverse 遥操作**
  - 通过 Inverse 进行机器人遥操作控制

通过不同的遥操作设备控制机械臂与灵巧手，完成目标任务的动作示教。

### 2. 机器人数据采集

当前主要适配的 VLA 模型为 OpenPI π0.5。

采集程序采用 LeRobot v3 数据格式，记录模型训练所需的多模态数据，主要包括：

- 机械臂关节状态与目标动作
- 灵巧手状态与控制指令
- 多视角相机图像
- 任务语言指令
- 时间戳及 Episode 信息

采集完成后，将数据交由 OpenPI 项目进行后续处理。

---

## 二、数据处理与模型训练

> 本阶段由当前 OpenPI CR3/O6 项目完成。

### 3. 数据格式转换

由于当前采集程序输出的是 LeRobot v3 格式，而本项目使用的 OpenPI 训练链路基于 LeRobot v2.1，因此需要进行数据格式转换。

使用以下脚本：

`examples/cr3_o6/convert_data_to_lerobot.py`

将采集得到的 LeRobot v3 数据转换为符合当前训练配置要求的 LeRobot v2.1 数据集。

转换完成后，使用：

`examples/cr3_o6/check_dataset.py`

检查数据结构、关节顺序、动作维度及图像字段，确保数据符合模型训练要求。

### 4. 数据归一化

完成数据格式转换后，需要计算训练数据的归一化统计量。

使用以下脚本：

`scripts/compute_norm_stats.py`

统计数据集中状态和动作等数据的分布，生成对应的 `norm_stats.json`。

这些统计量用于训练和推理阶段的数据归一化及反归一化，保证模型输入输出的数据尺度一致。

### 5. 模型训练

将转换后的 LeRobot v2.1 数据集存放到指定的 Dataset 路径，并在训练配置中指定对应的数据集。

当前使用 OpenPI π0.5 模型，通过 LoRA 微调适配 CR3 机械臂与 O6 灵巧手。

主要训练步骤：

1. 确认训练配置与数据集路径。
2. 计算并检查数据归一化统计量。
3. 配置模型、Batch Size、学习率和训练步数。
4. 启动训练并定期保存 Checkpoint。
5. 根据训练结果选择合适的 Checkpoint 进行部署。

---

## 三、模型部署与实机执行

### 6. 模型部署

训练完成后，通过 `serve_policy.py` 加载指定的模型 Checkpoint，在本地或远程 GPU 服务器上启动 WebSocket 推理服务。

随后，使用 `deploy` 目录中的部署脚本连接模型服务器、机械臂及灵巧手。

部署系统主要执行以下流程：

1. 获取机械臂与灵巧手的实时状态。
2. 采集多视角相机图像。
3. 结合任务语言指令构建模型输入。
4. 通过 WebSocket 请求模型推理。
5. 获取模型预测的关节动作序列。
6. 通过机器人控制接口执行预测动作。

当前通过 `main_rtc.py` 实现异步推理与动作块调度，并提供相机预览和实机运动确认模式。

 


## 环境要求

训练环境与机器人 Python 环境相互独立：OpenPI 使用 Python 3.11 + `uv`，实机客户端使用
Python 3.12 虚拟环境。

## 克隆与初始化

```bash
git clone --branch cr5-o6-pi05 https://github.com/euiiue/openpi-for-diversity-robot.git openpi_cr3_o6
cd openpi_cr3_o6
uv sync
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"
```

执行 `env.sh` 后，所有默认可写目录都会从当前检出目录推导：`data/`、`assets/`、
`checkpoints/`、`.cache/jax/` 和部署配置目录。若存储位置不同，请在执行该文件前覆盖相应变量。

## 1. 转换已完成的 LeRobot v3 会话
  此阶段为数据转换阶段，主要执行过程中需要修改的部分为 "cr3_o6_ceshi_reviewed_20260912"字段

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"

export CR3_O6_DATASET_ID="local/cr3_o6_ceshi_reviewed_20260912"
export CR3_O6_DATASET_ROOT="$HF_LEROBOT_HOME/$CR3_O6_DATASET_ID"
export CR3_O6_SOURCE_ROOT_1="$OPENPI_DATA_HOME/raw/session_001"
export CR3_O6_SOURCE_ROOT_2="$OPENPI_DATA_HOME/raw/session_002"

uv run examples/cr3_o6/convert_data_to_lerobot.py \
  --source-roots "$CR3_O6_SOURCE_ROOT_1" "$CR3_O6_SOURCE_ROOT_2" \
  --repo-id "$CR3_O6_DATASET_ID" \
  --output-root "$CR3_O6_DATASET_ROOT"

uv run examples/cr3_o6/check_dataset.py \
  --root "$CR3_O6_DATASET_ROOT" \
  --repo-id "$CR3_O6_DATASET_ID"
```

仓库内的训练配置使用上述数据集标识。若有意改用其他标识，请在计算统计量和训练之前，修改
`pi05_cr3_o6_joint_abs_lora` 中对应的 `repo_id`。

## 2. 计算归一化统计量并训练

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"

export CONFIG_NAME=pi05_cr3_o6_joint_abs_lora
export EXP_NAME=cr3_o6_run

uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
uv run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME"
```

当前 CR3/O6 基线将 JAX 模型 `dtype` 设置为 `float32`，因此既不是纯 FP16 训练，也不是
FP16 混合精度训练。学习率在预热 500 步后达到 `2.5e-5`，并在第 45,000 步衰减至
`2.5e-6`。配置中的冻结过滤规则会冻结 PaliGemma 的图像分支。

如需续训，请保持相同的 `CONFIG_NAME` 和 `EXP_NAME`，并增加 `--resume`：

```bash
uv run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" --resume
```

## 3. 启动已训练检查点的策略服务

从实验目录选择已完成的检查点步数，再启动 WebSocket 策略服务：

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"

export CONFIG_NAME=pi05_cr3_o6_joint_abs_lora
export EXP_NAME=cr3_o6_run
export CHECKPOINT_STEP=45000
export POLICY_DIR="$OPENPI_CHECKPOINT_DIR/$CONFIG_NAME/$EXP_NAME/$CHECKPOINT_STEP"

uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config "$CONFIG_NAME" \
  --policy.dir "$POLICY_DIR"
```

## 4. 配置并运行可选实机客户端

先从安全模板创建本地配置副本。这些文件会被 Git 忽略，必须填写本机的控制器地址、O6 串口设备、
相机序列号、经现场审核的工作空间边界和运动限制。

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"

cp "$CR3_O6_CONFIG_DIR/robot.example.yaml" "$CR3_O6_CONFIG_DIR/robot.local.yaml"
cp "$CR3_O6_CONFIG_DIR/camera.example.yaml" "$CR3_O6_CONFIG_DIR/camera.local.yaml"
```

创建部署环境，并安装实机依赖和 WebSocket 客户端包：

```bash
"$CR3_O6_DEPLOY_PYTHON" -m venv .venv-cr3-deploy
export CR3_O6_DEPLOY_PYTHON="$OPENPI_ROOT/.venv-cr3-deploy/bin/python"
"$CR3_O6_DEPLOY_PYTHON" -m pip install --upgrade pip
"$CR3_O6_DEPLOY_PYTHON" -m pip install -r examples/cr3_o6/deploy/requirements.txt
"$CR3_O6_DEPLOY_PYTHON" -m pip install -e packages/openpi-client
```

将任务文本设为转换后数据集使用的任务文本。请先使用仅预览模式：它会连接相机和策略服务，但会在
下发任何机器人运动命令之前丢弃全部模型动作。

```bash
export CR3_O6_TASK_PROMPT="task text used by the converted dataset"

"$CR3_O6_DEPLOY_PYTHON" examples/cr3_o6/deploy/main_rtc.py \
  --host 127.0.0.1 --port 8000 \
  --config-dir "$CR3_O6_CONFIG_DIR" \
  --prompt "$CR3_O6_TASK_PROMPT" \
  --preview-port 8080 --preview-only
```

只有在审核本地配置、实时相机预览、机器人状态和工作空间之后，操作员才可使用
`--confirm-motion`。客户端在启动 ServoJ 控制前会刻意要求这一显式确认。

```bash
"$CR3_O6_DEPLOY_PYTHON" examples/cr3_o6/deploy/main_rtc.py \
  --host 127.0.0.1 --port 8000 \
  --config-dir "$CR3_O6_CONFIG_DIR" \
  --prompt "$CR3_O6_TASK_PROMPT" \
  --confirm-motion
```

## 验证

无需连接机器人、O6 灵巧手或相机，即可运行以下基线测试套件：

```bash
env -u PYTHONPATH uv run pytest \
  examples/cr3_o6/tests/test_portable_release.py \
  examples/cr3_o6/tests/test_joint_contract.py \
  examples/cr3_o6/tests/test_rtc_buffering.py \
  examples/cr3_o6/tests/test_camera_preview.py -q
```

预览测试会启动回环 HTTP 服务。`env -u PYTHONPATH` 在变量未设置时无副作用，并可避免 shell
注入的无关 Python 环境加载外部 pytest 插件。
