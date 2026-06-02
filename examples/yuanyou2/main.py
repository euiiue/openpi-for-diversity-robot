#!/usr/bin/env python3
"""
Main entry point for running Yuanyou2 robot with OpenPI policy.
Features agilex-level control improvements (interpolation, temporal sync, JPEG pipeline).

Usage:
    # Start policy server first:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_yuanyou2_lora_finetune \
        --policy.dir=checkpoints/pi05_yuanyou2_lora_finetune/my_experiment/20000

    # Then run this script on the robot side:
    python examples/yuanyou2/main.py --remote-host 127.0.0.1

    # With higher interpolation smoothness:
    python examples/yuanyou2/main.py --remote-host 127.0.0.1 --interpolation-hz 300
"""

import dataclasses
import logging
import signal

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

try:
    from examples.yuanyou2 import env as _env
except ImportError:
    import env as _env


@dataclasses.dataclass
class Args:
    """Command-line arguments for Yuanyou2 deployment."""

    # ---- Policy server connection ----
    remote_host: str = "127.0.0.1"
    """IP address of the policy server."""

    remote_port: int = 8000
    """Port of the policy server."""

    # ---- Control ----
    control_frequency: float = 10.0
    """Control loop frequency in Hz (policy query rate)."""

    action_horizon: int = 30
    """Number of actions in each chunk returned by policy."""

    open_loop_horizon: int = 10
    """Number of actions to execute before querying policy again."""

    # ---- Interpolation (agilex feature) ----
    interpolation_hz: float = 200.0
    """Frequency of the joint interpolation control thread."""

    preemptive_publishing: bool = True
    """New actions immediately redirect interpolation toward the new target."""

    # ---- Image preprocessing (agilex feature) ----
    use_jpeg_pipeline: bool = True
    """Apply JPEG compression/decompression to match training data distribution."""

    jpeg_quality: int = 95
    """JPEG quality for the compression pipeline."""

    obs_history_num: int = 1
    """Number of historical observations to include (for future multi-frame support)."""

    # ---- Action chunking ----
    chunk_size: int = 50
    """Size of the action chunk buffer in the interface."""

    # ---- RTC guidance (agilex feature; requires server-side policy support) ----
    use_rtc_guidance: bool = False
    """Enable Real-Time Chunking guidance for temporally consistent inference.

    When enabled, the previous action chunk is included in the observation
    sent to the policy server. The server-side policy must support RTC
    (via infer_with_rtc_guidance) for this to take effect. If the server
    doesn't support RTC, the extra fields are silently ignored.
    """

    # ---- Task ----
    prompt: str = "pick a cube and place it on another cube"
    """Language instruction for the robot."""

    num_episodes: int = 100
    """Number of episodes to run."""

    max_episode_steps: int = 250
    """Maximum steps per episode."""

    sensor_timeout: float = 30.0
    """Seconds to wait for the first complete ROS observation."""

    # ---- ROS topics ----
    head_image_topic: str = "/head_camera/usb_cam/image_raw"
    """ROS image topic for the head camera."""

    left_wrist_image_topic: str = "/left_wrist_d435/color/image_raw"
    """ROS image topic for the left wrist camera."""

    right_wrist_image_topic: str = "/right_wrist_d435/color/image_raw"
    """ROS image topic for the right wrist camera."""


def main(args: Args) -> None:
    logging.info("=" * 60)
    logging.info("Yuanyou2 OpenPI Deployment (agilex-enhanced)")
    logging.info("=" * 60)
    logging.info("Policy server: ws://%s:%d", args.remote_host, args.remote_port)
    logging.info("Control frequency: %.0f Hz (policy), %.0f Hz (interpolation)",
                 args.control_frequency, args.interpolation_hz)
    logging.info("Action horizon: %d, Open-loop horizon: %d",
                 args.action_horizon, args.open_loop_horizon)
    logging.info("JPEG pipeline: %s, Preemptive: %s, RTC: %s",
                 args.use_jpeg_pipeline, args.preemptive_publishing, args.use_rtc_guidance)
    logging.info("Prompt: '%s'", args.prompt)
    logging.info("=" * 60)

    if args.open_loop_horizon > args.action_horizon:
        logging.warning(
            "open_loop_horizon (%d) > action_horizon (%d). "
            "The policy may be queried before the previous chunk is exhausted.",
            args.open_loop_horizon, args.action_horizon,
        )

    # Connect to policy server.
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )

    metadata = ws_client_policy.get_server_metadata()
    logging.info("Connected to policy server. Metadata: %s", metadata)

    # Create environment with agilex-level features.
    environment = _env.Yuanyou2Environment(
        prompt=args.prompt,
        sensor_timeout=args.sensor_timeout,
        image_topics={
            "head": args.head_image_topic,
            "left_wrist": args.left_wrist_image_topic,
            "right_wrist": args.right_wrist_image_topic,
        },
        interpolation_hz=args.interpolation_hz,
        preemptive_publishing=args.preemptive_publishing,
        use_jpeg_pipeline=args.use_jpeg_pipeline,
        jpeg_quality=args.jpeg_quality,
        obs_history_num=args.obs_history_num,
        chunk_size=args.chunk_size,
        use_rtc_guidance=args.use_rtc_guidance,
    )

    # Standard policy agent with action chunk brokering.
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.open_loop_horizon,
        )
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],
        max_hz=args.control_frequency,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # Handle clean shutdown.
    def _signal_handler(signum, frame):
        logging.info("Received signal %d, stopping...", signum)
        environment.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logging.info("Starting Yuanyou2 robot control loop...")
    logging.info("Press Ctrl+C to stop.")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("Stopping robot. Ctrl+C pressed.")
    finally:
        environment.stop()
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    args: Args = tyro.cli(Args)
    main(args)
