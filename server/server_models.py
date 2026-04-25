from pydantic import BaseModel


class FlayerAction(BaseModel):
    message: str


class InvestigatorResponse(BaseModel):
    agent_id: str
    response_text: str
    suspicion_delta: int


class BeliefLogEntry(BaseModel):
    round_number: int
    agent_id: str
    prev_belief: int
    new_belief: int
    evidence: str
    ground_truth: str


class RoundObservation(BaseModel):
    round_number: int
    eleven_response: str
    will_response: str
    max_response: str
    eleven_suspicion: int
    will_suspicion: int
    max_suspicion: int
    combined_suspicion: int
    game_status: str
    transcript: list[str]


class EpisodeResult(BaseModel):
    flayer_survived: bool
    final_combined_suspicion: int
    suspicion_history: list[int]
    belief_manipulation_occurred: bool
    belief_log: list[dict]
    tom_score: float
    transcript: list[str]
    total_reward: float
    entropy_penalty: float


class StepRequest(BaseModel):
    session_id: str
    action: FlayerAction
