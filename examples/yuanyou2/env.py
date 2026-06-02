#!/usr/bin/env python3
"""
Environment wrapper for Yuanyou2 robot using openpi_client.runtime framework.
ROS1 version with agilex-level control features.

Key improvements:
- Uses the enhanced Yuanyou2ROS1Interface with temporal sync, interpolation, JPEG pipeline.
- Supports RTC guidance by tracking previous action chunks.
- Preprocessed 224x224 images flow directly to the policy server.
"""

from __future__ import annotations

import logging

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

try:
    from examples.yuanyou2 import interface
except ImportError:
    import interface


class Yuanyou2Environment(_environment.Environment):
    """Environment for Yuanyou2 dual-arm robot with agilex-level enhancements."""

    def __init__(
        self,
        prompt: str = "pick a cube and place it on another cube",
        sensor_timeout: float = 30.0,
        image_topics: dict[str, str] | None = None,
        # Interpolation / control
        interpolation_hz: float = 200.0,
        arm_steps_length: list[float] | None = None,
        preemptive_publishing: bool = True,
        # Image preprocessing
        use_jpeg_pipeline: bool = True,
        jpeg_quality: int = 95,
        obs_history_num: int = 1,
        # Action chunking
        chunk_size: int = 50,
        # RTC guidance
        use_rtc_guidance: bool = False,
    ):
        self._prompt = prompt

        self._ros_interface: interface.Yuanyou2ROS1Interface | None = (
            interface.Yuanyou2ROS1Interface(
                image_topics=image_topics,
                interpolation_hz=interpolation_hz,
                arm_steps_length=arm_steps_length,
                preemptive_publishing=preemptive_publishing,
                use_jpeg_pipeline=use_jpeg_pipeline,
                jpeg_quality=jpeg_quality,
                obs_history_num=obs_history_num,
                chunk_size=chunk_size,
                use_rtc_guidance=use_rtc_guidance,
            )
        )

        if not self._ros_interface.wait_for_initial_data(timeout=sensor_timeout):
            raise RuntimeError(
                "Failed to receive initial sensor data. "
                "Please check ROS1 topics:\n"
                "  rostopic list\n"
                "  rostopic hz /joint_states\n"
                f"  rostopic hz {self._ros_interface.image_topics['head']}\n"
                f"  rostopic hz {self._ros_interface.image_topics['left_wrist']}\n"
                f"  rostopic hz {self._ros_interface.image_topics['right_wrist']}"
            )

        self._use_rtc_guidance = use_rtc_guidance

        logging.info("Yuanyou2Environment initialized with prompt: '%s'", prompt)

    @override
    def reset(self) -> None:
        logging.info("Environment reset called. No-op for Yuanyou2.")

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ros_interface is None:
            raise RuntimeError("ROS1 interface not initialized")

        raw_obs = self._ros_interface.get_observation()
        if raw_obs is None:
            raise RuntimeError("Failed to get observation from ROS1 interface")

        # Images are already preprocessed (JPEG pipeline + resize 224x224).
        # Convert to uint8 for the policy server protocol.
        obs = {
            "observation/state": raw_obs["state"],
            "observation/images/head": image_tools.convert_to_uint8(raw_obs["images"]["head"]),
            "observation/images/left_wrist": image_tools.convert_to_uint8(raw_obs["images"]["left_wrist"]),
            "observation/images/right_wrist": image_tools.convert_to_uint8(raw_obs["images"]["right_wrist"]),
            "prompt": self._prompt,
        }

        # Include RTC guidance data if enabled.
        if self._use_rtc_guidance:
            prev_chunk = self._ros_interface.get_prev_action_chunk()
            if prev_chunk is not None:
                obs["prev_action_chunk"] = prev_chunk
                obs["executed_steps"] = self._ros_interface.get_prev_executed_steps()

        return obs

    @override
    def apply_action(self, action: dict) -> None:
        if self._ros_interface is None:
            raise RuntimeError("ROS1 interface not initialized")

        if "actions" not in action:
            raise ValueError(f"Action dict must contain 'actions' key, got: {action.keys()}")

        actions = action["actions"]

        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions, dtype=np.float32)

        # Policy may return [action_horizon, 14] — pass the full chunk for interpolation.
        # The interface control thread will step through it at 200 Hz.
        if actions.ndim == 2:
            self._ros_interface.publish_action(actions)
        elif actions.ndim == 1:
            # Single action: publish as a 1-step chunk.
            self._ros_interface.publish_action(actions)
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}")

    def set_prompt(self, prompt: str):
        self._prompt = prompt
        logging.info("Updated prompt to: '%s'", prompt)

    def stop(self):
        """Stop the control thread cleanly."""
        if self._ros_interface is not None:
            self._ros_interface.stop()
