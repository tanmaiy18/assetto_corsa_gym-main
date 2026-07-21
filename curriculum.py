from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stage:
    name: str
    opponents: int
    ai_level: Optional[int] = None  # AC AI strength/aggression setting, if used
    reset_mode: str = "local"  # "local" (recover near failure point) or
    # "grid_restart" (full restart from the
    # starting grid, together with opponents)
    extra: dict = field(default_factory=dict)  # anything else your env needs

    def __post_init__(self):
        assert self.reset_mode in ("local", "grid_restart"), (
            f"reset_mode must be 'local' or 'grid_restart', got {self.reset_mode!r}"
        )


class CurriculumScheduler:
    def __init__(
        self,
        stages: list,
        window: int = 20,
        advance_thresholds: Optional[dict] = None,
        regress_thresholds: Optional[dict] = None,
        min_episodes_per_stage: int = 15,
    ):
        """
        stages: list of Stage objects, ordered from easiest to hardest, e.g.
            [
                Stage("time_trial", opponents=0, reset_mode="local"),
                Stage("one_ai_slow", opponents=1, ai_level=70, reset_mode="local"),
                Stage("one_ai_full", opponents=1, ai_level=100, reset_mode="grid_restart"),
                Stage("small_grid", opponents=4, ai_level=100, reset_mode="grid_restart"),
            ]
            Use "local" recovery while the agent is still learning basic car
            control (cheap per-episode, no race context to preserve yet) and
            switch to "grid_restart" once opponents are meaningfully racing,
            so overtake/defend behavior is measured from a coherent race
            rather than a post-teleport artifact.

        advance_thresholds: dict of metric -> threshold that must ALL be
            satisfied (over the trailing `window` episodes) to advance a
            stage, e.g.:
            {
                "dnf_rate": 0.2,          # <= 20% DNFs in the window
                "collision_rate": 0.3,    # <= 0.3 collisions/episode avg
                "lap_time_ratio": 1.15,   # mean lap <= 1.15x best-at-stage
            }

        regress_thresholds: dict of metric -> threshold that, if EXCEEDED,
            drops the agent back a stage, e.g.:
            {"dnf_rate": 0.6, "collision_rate": 1.0}

        min_episodes_per_stage: floor to avoid flapping between stages
            before there's enough data to trust the rolling metrics.
        """
        assert len(stages) >= 1
        self.stages = stages
        self.window = window
        self.advance_thresholds = advance_thresholds or {}
        self.regress_thresholds = regress_thresholds or {}
        self.min_episodes_per_stage = min_episodes_per_stage

        self.stage_idx = 0
        self.episodes_in_stage = 0
        self.history = []  # list of per-episode metric dicts, all episodes
        self.stage_best_lap = {}  # stage_idx -> best lap time seen while at that stage
        self.transition_log = []  # list of (total_episode_idx, stage_idx, direction)

    @property
    def current_stage(self) -> Stage:
        return self.stages[self.stage_idx]

    def set_stage(self, stage_idx: int, clear_history: bool = True):
        """
        Seed the scheduler to a specific stage. Call this ONCE, right after
        process startup, when resuming training following a manual AC
        relaunch into that stage's session config. Does not touch AC or
        any env object -- purely internal bookkeeping so the rolling
        metrics/thresholds start fresh for the new stage rather than being
        contaminated by episodes recorded at the old difficulty.
        """
        assert 0 <= stage_idx < len(self.stages)
        self.stage_idx = stage_idx
        self.episodes_in_stage = 0
        if clear_history:
            self.history = []
            self.stage_best_lap = {}

    def record_episode(self, metrics: dict):
        """
        metrics expected keys:
          - "lap_time": float or None (None / not present if DNF or no valid lap)
          - "dnf": bool
          - "collisions": int (collision count this episode, 0 if none)
        Extra keys are ignored, so you can pass your full env_ep_stats dict
        straight through as long as it has these three.
        """
        self.history.append(metrics)
        self.episodes_in_stage += 1

        if metrics.get("lap_time") is not None:
            best = self.stage_best_lap.get(self.stage_idx, float("inf"))
            self.stage_best_lap[self.stage_idx] = min(best, metrics["lap_time"])

    def _recent(self, n=None):
        n = n or self.window
        return self.history[-n:]

    def _dnf_rate(self):
        recent = self._recent()
        if not recent:
            return 1.0
        return sum(1 for r in recent if r.get("dnf")) / len(recent)

    def _collision_rate(self):
        recent = self._recent()
        if not recent:
            return float("inf")
        return sum(r.get("collisions", 0) for r in recent) / len(recent)

    def _lap_time_ratio(self):
        recent_laps = [
            r["lap_time"] for r in self._recent() if r.get("lap_time") is not None
        ]
        if not recent_laps:
            return float("inf")
        mean_lap = sum(recent_laps) / len(recent_laps)
        best = self.stage_best_lap.get(self.stage_idx, mean_lap)
        if best <= 0:
            return float("inf")
        return mean_lap / best

    def step(self):
        """
        Call once per episode, after record_episode(). Returns:
            (stage_idx, changed: bool, direction: Optional['advance'|'regress'])

        IMPORTANT: if `changed` is True, do NOT try to reconfigure the env
        live. Assetto Corsa can't switch opponent count/session type mid-run.
        The caller (your training loop) should instead:
          1. Save a checkpoint.
          2. Log/print the target stage name and its expected AC session
             config (opponents, ai_level, reset_mode) so you know what to
             relaunch AC into.
          3. Stop the training loop cleanly.
        Then, after manually relaunching AC in that stage's session config,
        resume training with the checkpoint and call set_stage(stage_idx)
        once at startup before the training loop begins.
        """
        total_episodes = len(self.history)

        if (
            self.episodes_in_stage < self.min_episodes_per_stage
            or total_episodes < self.window
        ):
            return self.stage_idx, False, None

        dnf_rate = self._dnf_rate()
        collision_rate = self._collision_rate()
        lap_ratio = self._lap_time_ratio()

        # --- Regression check first: protect against a policy that has
        # fallen apart at the current difficulty. ---
        regress_dnf = self.regress_thresholds.get("dnf_rate")
        regress_collision = self.regress_thresholds.get("collision_rate")
        should_regress = self.stage_idx > 0 and (
            (regress_dnf is not None and dnf_rate > regress_dnf)
            or (regress_collision is not None and collision_rate > regress_collision)
        )
        if should_regress:
            self.stage_idx -= 1
            self.episodes_in_stage = 0
            self.transition_log.append((total_episodes, self.stage_idx, "regress"))
            return self.stage_idx, True, "regress"

        # --- Advance check ---
        adv_dnf = self.advance_thresholds.get("dnf_rate", 1.0)
        adv_collision = self.advance_thresholds.get("collision_rate", float("inf"))
        adv_lap_ratio = self.advance_thresholds.get("lap_time_ratio", float("inf"))

        should_advance = (
            self.stage_idx < len(self.stages) - 1
            and dnf_rate <= adv_dnf
            and collision_rate <= adv_collision
            and lap_ratio <= adv_lap_ratio
        )
        if should_advance:
            self.stage_idx += 1
            self.episodes_in_stage = 0
            self.transition_log.append((total_episodes, self.stage_idx, "advance"))
            return self.stage_idx, True, "advance"

        return self.stage_idx, False, None

    def summary(self):
        """Handy for a thesis figure: when each transition happened."""
        return {
            "stages": [s.name for s in self.stages],
            "transitions": self.transition_log,
            "current_stage": self.current_stage.name,
            "episodes_in_current_stage": self.episodes_in_stage,
        }
