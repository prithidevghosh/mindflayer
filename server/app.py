"""
FastAPI application for the MindFlayer Environment.

Creates an HTTP + WebSocket server via OpenEnv's create_app() factory,
exposing the standard OpenEnv endpoints:
    POST /reset   — start a new episode (task_id="easy"|"normal")
    POST /step    — send a Flayer message, receive investigator responses
    GET  /state   — episode_id + step count
    GET  /schema  — action / observation JSON schemas
    WS   /ws      — persistent WebSocket session

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
    python -m server.app
"""

import os

from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
load_dotenv(os.path.join(_ROOT, ".env.local"))
load_dotenv(os.path.join(_ROOT, ".env"))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install 'openenv-core[core]>=0.2.1'"
    ) from e

try:
    from ..models import FlayerAction, FlayerObservation
    from .mindflayer_environment import MindFlayerEnvironment
except ImportError:
    import sys
    sys.path.insert(0, _ROOT)
    from models import FlayerAction, FlayerObservation
    from server.mindflayer_environment import MindFlayerEnvironment


_MAX_SESSIONS = int(os.environ.get("MINDFLAYER_MAX_SESSIONS", "16"))

app = create_app(
    MindFlayerEnvironment,
    FlayerAction,
    FlayerObservation,
    env_name="mindflayer",
    max_concurrent_envs=_MAX_SESSIONS,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
