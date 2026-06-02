import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def infer_with_rtc_guidance(
        self,
        obs: Dict,
        prev_action_chunk,
        executed_steps: int = 0,
        inference_delay: int = 4,
        execute_horizon: int = 1,
    ) -> Dict:
        """Infer actions with RTC (Real-Time Chunking) guidance.

        Ported from agilex openpi-agilex. Wraps the RTC parameters into the
        observation dict so the server-side policy can apply temporal guidance.

        Falls back to normal infer() if the server/policy doesn't support RTC.
        """
        import numpy as np

        obs = dict(obs)  # Don't mutate the caller's dict.
        obs["rtc_guidance"] = {
            "prev_action_chunk": np.asarray(prev_action_chunk),
            "executed_steps": executed_steps,
            "inference_delay": inference_delay,
            "execute_horizon": execute_horizon,
        }
        return self.infer(obs)

    @override
    def reset(self) -> None:
        pass
