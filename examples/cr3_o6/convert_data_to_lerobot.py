"""Convert explicitly selected, finalized GELLO joint sessions from LeRobot v3 to v2.1."""

from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path

import av
from lerobot.common.datasets import lerobot_dataset as dataset_module
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
import tyro

HAND_NAMES = ("thumb_flex", "thumb_yaw", "index_flex", "middle_flex", "ring_flex", "little_flex")
STATE_NAMES = [f"cr3.q{i}.rad" for i in range(1, 7)] + [f"o6.{name}.position" for name in HAND_NAMES]
ACTION_NAMES = [f"cr3.target_q{i}.rad" for i in range(1, 7)] + [f"o6.{name}.command" for name in HAND_NAMES]
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))


def main(
    source_roots: list[Path],
    repo_id: str = "local/cr3_o6_joint_absolute",
    output_root: Path | None = None,
    allow_needs_review: bool = False,
    drop_failures: bool = False,
    episode_indices: list[int] | None = None,
):
    output = (output_root or HF_LEROBOT_HOME / repo_id).expanduser().resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(f"Output or incomplete conversion already exists: {output}")
    if not source_roots:
        raise ValueError("Select at least one finalized joint session")
    sessions = [root.expanduser().resolve() for root in source_roots]
    if len(set(sessions)) != len(sessions):
        raise ValueError("Duplicate source sessions")
    features = {
        "observation.state": {"dtype": "float32", "shape": (12,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (12,), "names": ACTION_NAMES},
        **{
            key: {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channel"]}
            for key in IMAGE_KEYS
        },
    }
    selected = {}
    for root in sessions:
        selected[root] = []
        info = json.loads((root / "meta/info.json").read_text())
        if info["codebase_version"] != "v3.0" or info["fps"] != 20 or info["total_episodes"] < 1:
            raise ValueError(f"Expected finalized nonempty v3.0 / 20 Hz session: {root}")
        selected_indices = range(info["total_episodes"]) if episode_indices is None else episode_indices
        for index in selected_indices:
            if index < 0 or index >= info["total_episodes"]:
                raise ValueError(f"{root}, episode {index}: out of range")
            sidecar = json.loads((root / f"meta/collection/episode_{index:06d}.json").read_text())
            if drop_failures and sidecar["outcome"] in {"failure", "faliure"}:
                continue
            selected[root].append(index)
            if sidecar["recording_mode"] != "joint" or sidecar["outcome"] != "success":
                raise ValueError(f"{root}, episode {index}: requires successful joint demonstration")
            if sidecar["quality"]["needs_review"] and not allow_needs_review:
                raise ValueError(
                    f"{root}, episode {index}: needs_review; inspect collection evidence before conversion"
                )
        for key, expected in features.items():
            actual = info["features"][key]
            if key not in IMAGE_KEYS and (actual["shape"] != list(expected["shape"]) or actual["names"] != expected["names"]):
                raise ValueError(f"{root}: {key} joint names/units/shape mismatch")
    if not any(selected.values()):
        raise ValueError("No successful episodes remain")
    # Use the same encoding settings as the collector.
    original_encoder = dataset_module.encode_video_frames
    dataset_module.encode_video_frames = partial(original_encoder, vcodec="h264", pix_fmt="yuv420p", g=20, crf=28)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=building,
        fps=20,
        robot_type="cr3_o6",
        features=features,
        use_videos=True,
        image_writer_threads=0,
        video_backend="pyav",
    )
    counts = []
    try:
        for root in sessions:
            info = json.loads((root / "meta/info.json").read_text())
            episodes = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
            data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))]).sort_values(
                "index"
            )
            if len(data) != info["total_frames"] or len(episodes) != info["total_episodes"]:
                raise ValueError(f"{root}: metadata totals mismatch")
            if data["index"].tolist() != list(range(len(data))):
                raise ValueError(f"{root}: missing or duplicate frame indices")
            if sorted(episodes["episode_index"].tolist()) != list(range(len(episodes))):
                raise ValueError(f"{root}: invalid episode indices")
            total = 0
            converted_frames = 0
            mapping = []
            for _, ep in episodes.sort_values("episode_index").iterrows():
                rows = data[data["episode_index"] == ep["episode_index"]]
                length = int(ep["length"])
                if len(rows) != length or length == 0 or rows["frame_index"].tolist() != list(range(length)):
                    raise ValueError(f"{root}: invalid episode frames")
                total += length
                if int(ep["episode_index"]) not in selected[root]:
                    continue
                if not np.allclose(rows["timestamp"], np.arange(length) / 20, atol=1e-4, rtol=0):
                    raise ValueError(f"{root}: nonuniform 20 Hz timestamps")
                if len(ep["tasks"]) != 1 or not str(ep["tasks"][0]).strip():
                    raise ValueError(f"{root}: expected one nonempty task per episode")
                states = np.stack(rows["observation.state"]).astype(np.float32)
                actions = np.stack(rows["action"]).astype(np.float32)
                for values in (states, actions):
                    if values.shape != (length, 12) or not np.isfinite(values).all():
                        raise ValueError(f"{root}: invalid state/action")
                    if np.any((values[:, 6:] < 0) | (values[:, 6:] > 255)):
                        raise ValueError(f"{root}: O6 values outside 0..255")
                with ExitStack() as stack:
                    decoders = {}
                    starts = {}
                    for key in IMAGE_KEYS:
                        path = root / info["video_path"].format(
                            video_key=key,
                            chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                            file_index=int(ep[f"videos/{key}/file_index"]),
                        )
                        container = stack.enter_context(av.open(str(path)))
                        start = float(ep[f"videos/{key}/from_timestamp"])
                        stream = container.streams.video[0]
                        container.seek(int(start / stream.time_base), stream=stream, backward=True)
                        decoders[key] = container.decode(video=0)
                        starts[key] = start
                    for index in range(length):
                        images = {}
                        for key in IMAGE_KEYS:
                            frame = next(decoders[key])
                            if index == 0:
                                while float(frame.time) < starts[key] - 1e-4:
                                    frame = next(decoders[key])
                            if abs(float(frame.time) - starts[key] - index / 20) > 1e-3:
                                raise ValueError(f"{root}: video timestamp mismatch: {key}, frame {index}")
                            images[key] = frame.reformat(width=224, height=224, format="rgb24").to_ndarray()
                        dataset.add_frame(
                            {
                                **images,
                                "observation.state": states[index],
                                "action": actions[index],
                                "task": str(ep["tasks"][0]),
                            }
                        )
                mapping.append({"source_episode": int(ep["episode_index"]), "output_episode": dataset.meta.total_episodes})
                dataset.save_episode()
                converted_frames += length
            if total != len(data):
                raise ValueError(f"{root}: frames reference unknown episodes")
            counts.append({"source": str(root), "episodes": len(mapping), "frames": converted_frames,
                           "episode_mapping": mapping,
                           "excluded_episodes": sorted(set(range(len(episodes))) - set(selected[root]))})
    finally:
        dataset.stop_image_writer()
        dataset_module.encode_video_frames = original_encoder
    (building / "conversion.json").write_text(json.dumps(counts, indent=2, ensure_ascii=False))
    building.rename(output)
    print(f"Converted to {output}")


if __name__ == "__main__":
    tyro.cli(main)
