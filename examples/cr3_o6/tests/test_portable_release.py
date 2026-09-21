from pathlib import Path
import subprocess


def test_env_script_derives_repo_relative_paths_from_any_cwd(tmp_path):
    script = Path(__file__).resolve().parents[1] / "env.sh"
    command = (
        f'source "{script}"; '
        'printf "%s\\n%s\\n%s\\n%s\\n%s\\n" '
        '"$OPENPI_ROOT" "$HF_LEROBOT_HOME" "$OPENPI_ASSETS_DIR" '
        '"$OPENPI_CHECKPOINT_DIR" "$OPENPI_JAX_CACHE_DIR"'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    root = Path(__file__).resolve().parents[3]
    assert result.stdout.splitlines() == [
        str(root),
        str(root / "data" / "lerobot"),
        str(root / "assets"),
        str(root / "checkpoints"),
        str(root / ".cache" / "jax"),
    ]


def test_train_config_reads_output_locations_when_created(monkeypatch, tmp_path):
    import dataclasses

    from openpi.training import config

    assets = tmp_path / "assets"
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setenv("OPENPI_ASSETS_DIR", str(assets))
    monkeypatch.setenv("OPENPI_CHECKPOINT_DIR", str(checkpoints))

    train_config = config.TrainConfig(name="portable")
    assert train_config.assets_dirs == assets / "portable"
    resumed = dataclasses.replace(train_config, exp_name="run")
    assert resumed.checkpoint_dir == checkpoints / "portable" / "run"


def test_only_baseline_cr3_o6_config_is_registered():
    from openpi.training import config

    names = {item.name for item in config._CONFIGS}
    assert "pi05_cr3_o6_joint_abs_lora" in names
    removed = {
        f"pi05_cr3_o6_{'dagger'}_round1_lora",
        f"pi05_cr3_o6_{'final_fantasy'}_20260915_lora",
        f"pi05_cr3_o6_{'data_v21'}_lora",
        "pi05_dobot_cr5_o6_roi_lora",
    }
    assert names.isdisjoint(removed)
