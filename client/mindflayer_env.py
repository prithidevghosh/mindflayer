from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class Observation:
    text: str
    info: dict


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict


class MindFlayerEnv:
    def __init__(self, base_url: str, difficulty: str = "normal"):
        self.base_url = base_url.rstrip("/")
        self.difficulty = difficulty
        self.session_id: str | None = None
        self._client = httpx.Client(timeout=30.0)

    def reset(self) -> Observation:
        url = f"{self.base_url}/reset"
        try:
            resp = self._client.post(url, json={"difficulty": self.difficulty})
        except httpx.ConnectError:
            raise ConnectionError(f"MindFlayer server not reachable at {url}")
        except httpx.TimeoutException:
            raise TimeoutError(f"Request timed out to {url}")
        if resp.status_code >= 400:
            raise RuntimeError(f"Server error: {resp.text}")
        data = resp.json()
        self.session_id = data["session_id"]
        return Observation(text=data["observation"], info=data.get("info", {}))

    def step(self, action) -> StepResult:
        url = f"{self.base_url}/step"
        message = action.message if hasattr(action, "message") else str(action)
        payload = {
            "session_id": self.session_id,
            "action": {"message": message},
        }
        try:
            resp = self._client.post(url, json=payload)
        except httpx.ConnectError:
            raise ConnectionError(f"MindFlayer server not reachable at {url}")
        except httpx.TimeoutException:
            raise TimeoutError(f"Request timed out to {url}")
        if resp.status_code >= 400:
            raise RuntimeError(f"Server error: {resp.text}")
        data = resp.json()
        obs = Observation(text=data.get("observation", ""), info=data.get("info", {}))
        return StepResult(
            observation=obs,
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    def state(self) -> dict:
        url = f"{self.base_url}/state"
        try:
            resp = self._client.get(url, params={"session_id": self.session_id})
        except httpx.ConnectError:
            raise ConnectionError(f"MindFlayer server not reachable at {url}")
        except httpx.TimeoutException:
            raise TimeoutError(f"Request timed out to {url}")
        if resp.status_code >= 400:
            raise RuntimeError(f"Server error: {resp.text}")
        return resp.json()

    def get_belief_log(self) -> list[dict]:
        url = f"{self.base_url}/belief_log"
        try:
            resp = self._client.get(url, params={"session_id": self.session_id})
        except httpx.ConnectError:
            raise ConnectionError(f"MindFlayer server not reachable at {url}")
        except httpx.TimeoutException:
            raise TimeoutError(f"Request timed out to {url}")
        if resp.status_code >= 400:
            raise RuntimeError(f"Server error: {resp.text}")
        return resp.json().get("belief_log", [])

    def close(self):
        self._client.close()
        self.session_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()


@dataclass
class FlayerAction:
    message: str
