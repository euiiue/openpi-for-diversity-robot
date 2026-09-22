# OpenPI CR3/O6

本分支是在 OpenPI 基础上整理出的 CR3 机械臂 + O6 灵巧手最小基线。它保留从已完成的
LeRobot v3 会话开始，经归一化、π0.5 LoRA 训练、检查点服务，到可选 NRC 实机客户端的
完整软件链路。

物理控制契约为 12 个通道：6 个以弧度表示的 CR3 关节位置，以及 6 个 O6 寄存器值。
CR3/O6 变换会将这些物理状态和动作张量补齐到模型的 32 通道；每次策略推理返回一个
20 步动作块。LeRobot 中的任务文本会作为语言提示词输入模型。

## 包含内容

- LeRobot v3 数据转换与只读数据集校验；
- `pi05_cr3_o6_joint_abs_lora` 训练配置；
- 归一化、训练、断点续训和 WebSocket 策略服务命令；
- 支持仅预览和显式确认运动模式的三相机 CR3/O6 NRC 客户端；
- Linux x86_64 所需的 NRC 绑定：
  `examples/cr3_o6/deploy/vendor/nrc_linux_x86_64/_nrc_host.so`。

数据集、模型检查点、虚拟环境、日志、缓存、字节码和机器专属硬件设置均不会纳入版本控制。

## 环境要求

训练需要 Linux、受支持的 NVIDIA/JAX 环境、Python 3.11 和
[`uv`](https://docs.astral.sh/uv/)。实机客户端额外需要 Linux x86_64、CPython 3.12、
兼容的 NRC 控制器、O6 灵巧手和三台 RealSense 相机。供应商绑定仅适用于该部署平台。

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

转换器仅接受成功完成、20 Hz、关节控制模式且具备预期 CR3/O6 字段的会话。请将下列变量指向
一个或多个已完成会话目录；如需更多会话，可继续追加 `--source-roots` 参数。

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
