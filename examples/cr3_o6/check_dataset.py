"""Read-only check of a converted CR3 dataset through the actual training transforms."""

import json
from pathlib import Path

from convert_data_to_lerobot import ACTION_NAMES
from convert_data_to_lerobot import STATE_NAMES
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from openpi import transforms
from openpi.shared import normalize
from openpi.training import config


def main(root: Path, repo_id: str = "local/cr3_o6_joint_absolute"):
    cfg = config.get_config("pi05_cr3_o6_joint_abs_lora")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = LeRobotDataset(
        repo_id, root=root, delta_timestamps={"action": [i / 20 for i in range(20)]}, video_backend="pyav"
    )
    if dataset.meta.fps != 20:
        raise ValueError("CR3 deployment requires 20 Hz data")
    for key, names in (("observation.state", STATE_NAMES), ("action", ACTION_NAMES)):
        if dataset.meta.features[key]["names"] != names:
            raise ValueError(f"{key}: wrong joint order/units")
    pipeline = transforms.compose(
        [transforms.PromptFromLeRobotTask(dataset.meta.tasks), *dc.repack_transforms.inputs, *dc.data_transforms.inputs]
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    max_target_error = 0.0
    for index in range(len(dataset)):
        sample = pipeline(dataset[index])
        max_target_error = max(max_target_error, float(np.max(np.abs(sample["actions"][0, :6] - sample["state"][:6]))))
        for key in stats:
            stats[key].update(sample[key][None])
    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}
    model_pipeline = transforms.compose(
        [transforms.Normalize(norm_stats, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs]
    )
    for index in {0, len(dataset) // 2, len(dataset) - 1}:
        sample = model_pipeline(pipeline(dataset[index]))
        if sample["state"].shape != (32,) or sample["actions"].shape != (20, 32):
            raise ValueError("Unexpected padded model shapes")
        if not np.isfinite(sample["state"]).all() or not np.isfinite(sample["actions"]).all():
            raise ValueError("Nonfinite normalized model input")
    print(
        json.dumps(
            {
                "frames": len(dataset),
                "episodes": dataset.meta.total_episodes,
                "tasks": dataset.meta.tasks,
                "max_target_error_deg": np.rad2deg(max_target_error),
                "model_state_shape": [32],
                "model_action_shape": [20, 32],
                "note": "Format/transforms passed; this does not certify demonstration quality or robot motion.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)
