# OpenPI CR3/O6

This branch is a focused CR3 arm + O6 hand baseline on top of OpenPI. It keeps
the complete software path from finalized LeRobot v3 sessions through
normalization, π0.5 LoRA training, checkpoint serving, and an optional NRC
hardware client.

The physical contract is 12 channels: six CR3 joint positions in radians and
six O6 register values. The CR3/O6 transforms pad those physical state and
action tensors to the model's 32 channels; each policy result contains a
20-step action chunk. LeRobot task text is used as the language prompt.

## What is included

- LeRobot v3 conversion and read-only dataset validation;
- the `pi05_cr3_o6_joint_abs_lora` training configuration;
- normalization, training, resume, and WebSocket policy-service commands;
- a three-camera CR3/O6 NRC client with preview-only and motion-confirmation
  modes;
- the required Linux x86_64 NRC binding at
  `examples/cr3_o6/deploy/vendor/nrc_linux_x86_64/_nrc_host.so`.

Datasets, model checkpoints, virtual environments, logs, caches, bytecode, and
machine-specific hardware settings are intentionally not versioned.

## Requirements

Training requires Linux, a supported NVIDIA/JAX environment, Python 3.11, and
[`uv`](https://docs.astral.sh/uv/). The hardware client additionally requires
Linux x86_64, CPython 3.12, a compatible NRC controller, an O6 hand, and three
RealSense cameras. The vendor binding is specific to that deployment platform.

The normal training environment and the robot Python environment are separate:
use `uv` with Python 3.11 for OpenPI, and a Python 3.12 virtual environment for
the hardware client.

## Clone and initialize

```bash
git clone --branch cr5-o6-pi05 https://github.com/euiiue/openpi-for-diversity-robot.git openpi_cr3_o6
cd openpi_cr3_o6
uv sync
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"
```

Sourcing `env.sh` derives all default writable locations from the checkout:
`data/`, `assets/`, `checkpoints/`, `.cache/jax/`, and the deployment-config
directory. Override any of these before sourcing the file when storage lives
elsewhere.

## 1. Convert finalized LeRobot v3 sessions

The converter accepts only successful, 20 Hz, joint-mode sessions with the
expected CR3/O6 fields. Point the variables below at one or more finalized
session directories; use additional `--source-roots` arguments for more
sessions.

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

The checked-in training configuration uses the dataset identifier shown above.
If you deliberately choose another identifier, change that one `repo_id` in
`pi05_cr3_o6_joint_abs_lora` before computing statistics and training.

## 2. Compute normalization statistics and train

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"

export CONFIG_NAME=pi05_cr3_o6_joint_abs_lora
export EXP_NAME=cr3_o6_run

uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
uv run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME"
```

The current CR3/O6 baseline sets the JAX model `dtype` to `float32`; it is
therefore neither pure FP16 nor FP16 mixed-precision training. Its learning-rate
schedule peaks at `2.5e-5` after 500 warmup steps and decays to `2.5e-6` by
step 45,000. It freezes the PaliGemma image branch as defined by the
configuration's freeze filter.

To continue an existing experiment, preserve the same `CONFIG_NAME` and
`EXP_NAME` and add `--resume`:

```bash
uv run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" --resume
```

## 3. Serve a trained checkpoint

Choose a completed checkpoint step from the experiment directory, then start
the WebSocket policy service:

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

## 4. Configure and run the optional hardware client

Create local copies of the safe templates. These files are ignored by Git and
must contain the controller address, O6 serial device, camera serial numbers,
site-reviewed workspace bounds, and motion limits for the specific robot.

```bash
source examples/cr3_o6/env.sh
cd "$OPENPI_ROOT"

cp "$CR3_O6_CONFIG_DIR/robot.example.yaml" "$CR3_O6_CONFIG_DIR/robot.local.yaml"
cp "$CR3_O6_CONFIG_DIR/camera.example.yaml" "$CR3_O6_CONFIG_DIR/camera.local.yaml"
```

Create the deployment environment and install both its hardware dependencies
and the WebSocket client package:

```bash
"$CR3_O6_DEPLOY_PYTHON" -m venv .venv-cr3-deploy
export CR3_O6_DEPLOY_PYTHON="$OPENPI_ROOT/.venv-cr3-deploy/bin/python"
"$CR3_O6_DEPLOY_PYTHON" -m pip install --upgrade pip
"$CR3_O6_DEPLOY_PYTHON" -m pip install -r examples/cr3_o6/deploy/requirements.txt
"$CR3_O6_DEPLOY_PYTHON" -m pip install -e packages/openpi-client
```

Set the task text to the task used in the converted dataset. Start with
preview-only mode: it connects cameras and the policy service but discards all
model actions before any robot motion command is issued.

```bash
export CR3_O6_TASK_PROMPT="task text used by the converted dataset"

"$CR3_O6_DEPLOY_PYTHON" examples/cr3_o6/deploy/main_rtc.py \
  --host 127.0.0.1 --port 8000 \
  --config-dir "$CR3_O6_CONFIG_DIR" \
  --prompt "$CR3_O6_TASK_PROMPT" \
  --preview-port 8080 --preview-only
```

Only after reviewing the local configuration, the live camera preview, robot
state, and workspace may an operator use `--confirm-motion`. It is intentionally
required by the client before ServoJ control can start.

```bash
"$CR3_O6_DEPLOY_PYTHON" examples/cr3_o6/deploy/main_rtc.py \
  --host 127.0.0.1 --port 8000 \
  --config-dir "$CR3_O6_CONFIG_DIR" \
  --prompt "$CR3_O6_TASK_PROMPT" \
  --confirm-motion
```

## Verification

Run the focused baseline suite without a robot, O6 hand, or camera connection:

```bash
env -u PYTHONPATH uv run pytest \
  examples/cr3_o6/tests/test_portable_release.py \
  examples/cr3_o6/tests/test_joint_contract.py \
  examples/cr3_o6/tests/test_rtc_buffering.py \
  examples/cr3_o6/tests/test_camera_preview.py -q
```

The preview tests use a loopback HTTP server. The `env -u PYTHONPATH` prefix is
harmless when unset and prevents a shell-injected, unrelated Python environment
from loading external pytest plugins.
