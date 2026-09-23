# OpenPI CR3/O6

本分支是在 OpenPI 基础上整理出的 CR3 机械臂 + O6 灵巧手的复现基线，后续适配CR5或者珞石机械臂都会按照此流程执行。

本项目执行流程大致为：
##1、控制机械臂采集数据：当前控制设备主要有gello 主臂遥操（此部分又分为6轴主臂和7轴主臂）、inexbot微雪主臂遥操（四自由度）以及inverse进行遥操控制。
##2、采集数据，目前主要适配的模型为openpi 05模型所以当前采集数据格式为openpi 官方要求lerobot格式。
--------------------- 以上流程都为在gello_CR项目完成内容-------------------------

##3、数据转换，通过当前数据采集程序采集的数据格式为lerobot V3格式，其并不符合openpi的数据准入流程，需要通过 `convert_data_to_lerobot.py/` 脚本进行数据转换将直接采集的lerobot V3格式数据转成lerobot v2.1格式
##4、数据归一化，在完成数据转成lerobot v2.1格式后，根据 transformer 架构的输入需求，输入数据需要运行 `compute_norm_stats.py/` 脚本统一进行归一化处理，把所有的数据统一转化成为0-1区间的定量方便后续量化估计处理。
##5、模型训练，将完成处理后的数据放到指定的dataset路径，开启模型训练
##6、模型部署，将训练好的模型进行本地启动模型以及机械臂设备，通过deploy脚本打通模型和机械臂的连接，让模型根据机械臂的state 和相机输入画面输出指定的关节数据。
----------------------以上部分为当前阶段模型部署所做的流程-------------------------

 


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
