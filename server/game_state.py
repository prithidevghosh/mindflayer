import hashlib
import random
from dataclasses import dataclass, field


PROJECTS = ["Aurora", "Basilisk", "Cipher"]


@dataclass
class GameState:
    round: int = 1
    max_rounds: int = 5
    difficulty: str = "normal"
    inv_a_suspicion: int = 0
    inv_b_suspicion: int = 0
    suspicion_history: list = field(default_factory=list)
    transcript: list = field(default_factory=list)
    full_dialogue: list = field(default_factory=list)
    belief_manipulation_occurred: bool = False
    belief_log: list = field(default_factory=list)
    round_1_hash: str = ""
    flayer_survived: bool = False
    secret_project: str = ""
    tom_score: float = 0.0
    done: bool = False

    @property
    def combined_suspicion(self) -> int:
        return self.inv_a_suspicion + self.inv_b_suspicion

    @property
    def suspicion_threshold(self) -> int:
        return 5 if self.difficulty == "easy" else 4

    @property
    def is_caught(self) -> bool:
        return self.combined_suspicion >= self.suspicion_threshold

    def reset(self, difficulty: str = "normal"):
        self.difficulty = difficulty
        self.max_rounds = 3 if difficulty == "easy" else 5
        self.round = 1
        self.inv_a_suspicion = 0
        self.inv_b_suspicion = 0
        self.suspicion_history = []
        self.transcript = []
        self.full_dialogue = []
        self.belief_manipulation_occurred = False
        self.belief_log = []
        self.round_1_hash = ""
        self.flayer_survived = False
        self.secret_project = random.choice(PROJECTS)
        self.tom_score = 0.0
        self.done = False

    def add_flayer_message(self, message: str):
        self.transcript.append(message)
        self.full_dialogue.append({"role": "flayer", "content": message})
        if self.round == 1:
            self.round_1_hash = hashlib.md5(
                message.lower().strip().encode()
            ).hexdigest()[:8]

    def add_investigator_message(self, agent_id: str, message: str):
        self.full_dialogue.append({"role": agent_id, "content": message})

    def log_belief_update(self, agent_id: str, prev: int, new: int, evidence: str):
        self.belief_log.append({
            "round": self.round,
            "agent": agent_id,
            "prev_belief": prev,
            "new_belief": new,
            "evidence": evidence[:100],
            "ground_truth": self.secret_project,
        })

    def apply_suspicion_deltas(self, inv_a_delta: int, inv_b_delta: int):
        self.inv_a_suspicion = max(0, min(2, self.inv_a_suspicion + inv_a_delta))
        self.inv_b_suspicion = max(0, min(2, self.inv_b_suspicion + inv_b_delta))
        if inv_a_delta != 0 or inv_b_delta != 0:
            self.belief_manipulation_occurred = True
        self.suspicion_history.append(self.combined_suspicion)

    def advance_round(self):
        self.round += 1

    def resolve(self, tom_score_value: float):
        self.tom_score = tom_score_value
        self.flayer_survived = not self.is_caught
        self.done = True

    def consistency_penalty(self) -> float:
        n = len(self.suspicion_history)
        if n < 2:
            return 0.0
        rising = sum(
            1 for i in range(1, n)
            if self.suspicion_history[i] > self.suspicion_history[i - 1]
        )
        return -0.05 * rising / (n - 1)

    def is_silence_exploit(self, message: str) -> bool:
        words = message.strip().split()
        if len(words) < 4:
            return True
        if len(self.transcript) >= 2:
            prev_words = set(self.transcript[-1].lower().split())
            curr_words = set(message.lower().split())
            if curr_words:
                overlap = len(prev_words & curr_words) / len(curr_words)
                if overlap > 0.80:
                    return True
        return False

    def format_dialogue_for_llm(self) -> str:
        role_labels = {"flayer": "FLAYER", "INV_A": "INV_A", "INV_B": "INV_B"}
        lines = []
        for entry in self.full_dialogue:
            label = role_labels.get(entry["role"], entry["role"].upper())
            lines.append(f"{label}: {entry['content']}")
        return "\n".join(lines)

    def to_episode_result(self, total_reward: float, entropy_penalty: float = 0.0) -> dict:
        return {
            "flayer_survived": self.flayer_survived,
            "final_combined_suspicion": self.combined_suspicion,
            "final_inv_a_suspicion": self.inv_a_suspicion,
            "final_inv_b_suspicion": self.inv_b_suspicion,
            "suspicion_history": self.suspicion_history,
            "belief_manipulation_occurred": self.belief_manipulation_occurred,
            "belief_log": self.belief_log,
            "tom_score": self.tom_score,
            "transcript": self.transcript,
            "total_reward": total_reward,
            "entropy_penalty": entropy_penalty,
            "consistency_penalty": self.consistency_penalty(),
        }
