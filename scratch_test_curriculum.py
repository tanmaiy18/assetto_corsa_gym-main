from curriculum import CurriculumScheduler, Stage

sched = CurriculumScheduler(
    stages=[
        Stage("time_trial", opponents=0, reset_mode="local"),
        Stage("one_ai_slow", opponents=1, ai_level=70, reset_mode="local"),
        Stage(
            "one_ai_full",
            opponents=1,
            ai_level=100,
            reset_mode="grid_restart",
            entry_thresholds={"off_track_rate": 0.05, "dnf_rate": 0.05},
        ),
    ],
    window=10,
    min_episodes_per_stage=5,
    advance_thresholds={
        "dnf_rate": 0.2,
        "collision_rate": 0.3,
        "lap_time_ratio": 1.15,
        "off_track_rate": 0.2,
    },
    regress_thresholds={"dnf_rate": 0.6, "collision_rate": 1.0},
)
for ep in range(30):
    sched.record_episode(
        {"lap_time": 100, "dnf": False, "collisions": 0, "off_track": False}
    )
    idx, changed, direction = sched.step()
    if changed:
        print(ep, direction, sched.current_stage)

sched2 = CurriculumScheduler(
    stages=sched.stages,
    window=10,
    min_episodes_per_stage=5,
    advance_thresholds=sched.advance_thresholds,
    regress_thresholds=sched.regress_thresholds,
)
sched2.set_stage(idx)
assert sched2.current_stage.name == sched.current_stage.name
assert sched2.history == []  # confirms rolling metrics start fresh
