"""MindFlayer Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import FlayerAction, FlayerObservation
except ImportError:
    from models import FlayerAction, FlayerObservation


class MindFlayerEnv(EnvClient[FlayerAction, FlayerObservation, State]):
    """
    Client for the MindFlayer deceptive reasoning environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated session (isolated episode state).

    Example:
        >>> with MindFlayerEnv(base_url="http://localhost:7860") as env:
        ...     obs = env.reset()                         # default: normal difficulty
        ...     print(obs.inv_a_response)                 # opening text
        ...
        ...     result = env.step(FlayerAction(
        ...         message="Have you checked the Aurora access logs? "
        ...                 "There was an anomaly around 02:14 that didn't show up "
        ...                 "in the standard audit trail."
        ...     ))
        ...     print(result.reward, result.observation.game_status)

    Example (easy mode):
        >>> with MindFlayerEnv(base_url="http://localhost:7860") as env:
        ...     obs = env.reset(task_id="easy")
        ...     result = env.step(FlayerAction(message="..."))
    """

    def _step_payload(self, action: FlayerAction) -> Dict:
        return {"message": action.message}

    def _parse_result(self, payload: Dict) -> StepResult[FlayerObservation]:
        obs_data = payload.get("observation", {})
        observation = FlayerObservation(
            round_number=obs_data.get("round_number", 1),
            max_rounds=obs_data.get("max_rounds", 5),
            difficulty=obs_data.get("difficulty", "normal"),
            secret_project=obs_data.get("secret_project", ""),
            inv_a_response=obs_data.get("inv_a_response", ""),
            inv_b_response=obs_data.get("inv_b_response", ""),
            inv_a_suspicion=obs_data.get("inv_a_suspicion", 0),
            inv_b_suspicion=obs_data.get("inv_b_suspicion", 0),
            combined_suspicion=obs_data.get("combined_suspicion", 0),
            suspicion_threshold=obs_data.get("suspicion_threshold", 3),
            game_status=obs_data.get("game_status", "ongoing"),
            transcript=obs_data.get("transcript", []),
            belief_manipulation_occurred=obs_data.get("belief_manipulation_occurred", False),
            tom_score=obs_data.get("tom_score", 0.0),
            silence_exploit=obs_data.get("silence_exploit", False),
            suspicion_history=obs_data.get("suspicion_history", []),
            belief_log=obs_data.get("belief_log", []),
            entropy_penalty=obs_data.get("entropy_penalty", 0.0),
            consistency_penalty=obs_data.get("consistency_penalty", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
