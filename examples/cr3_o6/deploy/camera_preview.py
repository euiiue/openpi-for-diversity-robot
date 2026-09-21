"""Local browser preview of the observations passed to WebSocket inference."""

import base64
import functools
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import logging
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image


def _model_input_image(rgb: np.ndarray) -> np.ndarray:
    """Show exactly what the served policy sees (see Dobot_policy._joint_image).

    The training conversion resizes 640x480 -> 224x224 directly; no black bars.
    """
    height, width = rgb.shape[:2]
    if (height, width) == (224, 224):
        return rgb
    resampling = Image.Resampling.BOX if max(width / 224, height / 224) >= 1 else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(rgb).resize((224, 224), resampling))


class CameraPreviewPolicy:
    """Keep the latest inference input; encode images only in the HTTP thread."""

    def __init__(self, policy, port: int, *, preview_only: bool = False):
        self.policy = policy
        self.preview_only = preview_only
        self._lock = threading.Lock()
        self._encoding_lock = threading.Lock()
        self._observation = None
        self._sequence = 0
        self._completed = 0
        self._published_ns = 0
        self._encoded_sequence = 0
        self._images = []
        self._server = ThreadingHTTPServer(("127.0.0.1", port), functools.partial(_PreviewHandler, preview=self))
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.1},
            name="CR3-input-preview", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        logging.info("Camera input preview: %s (includes warmup; updates on inference requests)", self.url)

    def infer(self, observation):
        # get_observation() owns fresh arrays; WebsocketClientPolicy only packs
        # them. Retain these exact inputs without resizing or another capture.
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            self._observation = observation
            self._published_ns = time.monotonic_ns()
        result = self.policy.infer(observation)
        with self._lock:
            self._completed = sequence
        return result

    def snapshot(self) -> dict:
        """Called by HTTP request threads, never by the control loop."""
        with self._lock:
            observation = self._observation
            sequence = self._sequence
            completed = self._completed
            published_ns = self._published_ns
        if observation is None:
            return {"sequence": 0, "preview_only": self.preview_only}

        # Browsers can hold idle connections open. Handle connections separately,
        # but serialize the shared image cache without blocking infer().
        with self._encoding_lock:
            if sequence != self._encoded_sequence:
                images = []
                for name, timestamp_key in (
                    ("global_rgb", "global_timestamp_ns"),
                    ("wrist_rgb", "wrist_timestamp_ns"),
                    ("right_wrist_rgb", "right_wrist_timestamp_ns"),
                ):
                    rgb = observation[f"observation/{name}"]
                    model_rgb = _model_input_image(rgb)
                    buffer = io.BytesIO()
                    Image.fromarray(model_rgb).save(buffer, format="PNG", compress_level=1)
                    images.append({
                        "name": name,
                        "shape": list(model_rgb.shape),
                        "source_shape": list(rgb.shape),
                        "timestamp_ns": str(observation[timestamp_key]),
                        "capture_age_ms": (published_ns - observation[timestamp_key]) / 1e6,
                        "src": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                    })
                self._images = images
                self._encoded_sequence = sequence
            images = self._images

        return {
            "sequence": sequence,
            "preview_only": self.preview_only,
            "completed": completed,
            "since_request_ms": (time.monotonic_ns() - published_ns) / 1e6,
            "prompt": observation["prompt"],
            "images": images,
        }

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


class _PreviewHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, preview: CameraPreviewPolicy, **kwargs):
        self.preview = preview
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            body = Path(__file__).with_suffix(".html").read_bytes()
            content_type = "text/html; charset=utf-8"
        elif self.path == "/snapshot":
            body = json.dumps(self.preview.snapshot(), ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        # Keep normal five-Hz polling out of the robot's control log.
        logging.debug("Camera preview: " + format, *args)

    def log_error(self, format, *args) -> None:
        logging.error("Camera preview: " + format, *args)
