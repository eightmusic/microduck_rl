"""CPU cfg-invariant + state-machine regression tests for the jump family.

Builds cfgs and fake envs only — never steps a real simulation.

The family is THREE tasks on one shared recipe (microduck_jump_env_cfg):
  FrogJump  — continuous forward hopping (multi mode: every touchdown pays);
  LongJump  — one forward jump then stand (single mode);
  JumpHigh  — one vertical jump for height (single mode, gain-based rate).
"""

import math
import os

import pytest
import torch

import mjlab_microduck.tasks.mdp as microduck_mdp
import mjlab_microduck.tasks.microduck_jump_env_cfg as jump_cfg_mod
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    JUMP_AIR_TIME_CAP_S,
    JUMP_APEX_CEILING,
    JUMP_HIGH_APEX_CEILING,
    JUMP_TARGET_X,
    JUMP_TARGET_Z,
    STAND_Z,
    make_microduck_frog_jump_env_cfg,
    make_microduck_jump_high_env_cfg,
    make_microduck_long_jump_env_cfg,
    MicroduckFrogJumpRlCfg,
    MicroduckJumpHighRlCfg,
    MicroduckLongJumpRlCfg,
)

# (factory, task id, rl cfg, discriminator term, mode, episode length,
#  launch ceiling, drift tolerances). One row per family member.
FAMILY = {
    "frog": {
        "factory": make_microduck_frog_jump_env_cfg,
        "task_id": "Mjlab-FrogJump-Flat-MicroDuck",
        "rl_cfg": MicroduckFrogJumpRlCfg,
        "discriminator": "jump_distance",
        "target": JUMP_TARGET_X,
        "disc_weight": 8.0,
        "multi_flight": True,
        "episode_length_s": 6.0,
        "apex_ceiling": JUMP_APEX_CEILING,
        "yaw_tol": 0.8,
        "lat_tol": 0.05,
        "fwd_drift": False,
    },
    "long": {
        "factory": make_microduck_long_jump_env_cfg,
        "task_id": "Mjlab-LongJump-Flat-MicroDuck",
        "rl_cfg": MicroduckLongJumpRlCfg,
        "discriminator": "jump_distance",
        "target": JUMP_TARGET_X,
        "disc_weight": 8.0,
        "multi_flight": False,
        "episode_length_s": 4.0,
        "apex_ceiling": JUMP_APEX_CEILING,
        "yaw_tol": 0.5,
        "lat_tol": 0.03,
        "fwd_drift": False,
    },
    "high": {
        "factory": make_microduck_jump_high_env_cfg,
        "task_id": "Mjlab-JumpHigh-Flat-MicroDuck",
        "rl_cfg": MicroduckJumpHighRlCfg,
        "discriminator": "jump_height",
        "target": JUMP_TARGET_Z,
        "disc_weight": 20.0,
        "multi_flight": False,
        "episode_length_s": 5.0,
        "apex_ceiling": JUMP_HIGH_APEX_CEILING,
        "yaw_tol": None,  # drift taxes NOT registered: the verticality gate
        "lat_tol": None,  # inside the payment supersedes them (see factory)
        "fwd_drift": False,
    },
}
FAMILY_IDS = list(FAMILY)

# mjlab-style costs return >= 0 → negative weight (AGENTS.md sign convention).
# Drift taxes are per-task (asserted in test_drift_taxes_per_task).
BASE_MJLAB_COSTS = (
    "jump_stand_tax",
    "jump_crash_tax",
    "head_impact",
    "trunk_impact",
    "servo_acc_spike",
    "servo_stall",
    "action_rate_l2",
    "joint_torque_rate_l2",
    "self_collisions",
    "body_ang_vel",
    "angular_momentum",
)
# 0..1 task rewards → positive weight (discriminator asserted separately).
BASE_TASK_REWARDS = (
    "jump_air_time",
    "jump_launch",
    "jump_landing",
    "jump_landing_sharp",
)


# ── cfg-level invariants (every family member) ────────────────────────────────

@pytest.mark.parametrize("key", FAMILY_IDS)
def test_env_builds_train_and_play(key):
    spec = FAMILY[key]
    spec["factory"]()
    spec["factory"](play=True)


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_task_registered(key):
    import mjlab_microduck.tasks  # noqa: F401  (registers on import)
    from mjlab.tasks.registry import list_tasks

    assert FAMILY[key]["task_id"] in list_tasks()


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_task_rewards_present_and_discriminator_dominant(key):
    spec = FAMILY[key]
    cfg = spec["factory"]()
    for name in BASE_TASK_REWARDS:
        assert name in cfg.rewards, (key, name)
        assert cfg.rewards[name].weight > 0.0, (key, name)
    # exactly one discriminator per task, at its task-specific weight
    for disc in ("jump_distance", "jump_height"):
        if disc == spec["discriminator"]:
            assert cfg.rewards[disc].weight == spec["disc_weight"], key
            assert cfg.rewards[disc].params[("target_x" if disc == "jump_distance" else "target_z")] == spec["target"], key
        else:
            assert disc not in cfg.rewards, (key, disc)
    # the discriminator is THE task term: strictly larger than the flight
    # bootstrap. jump_launch (potential-based, nets zero per cycle) and the
    # per-step landing annuity are deliberately weighted on a different axis
    # (mass, not weight) — do NOT compare their weights here.
    assert cfg.rewards["jump_air_time"].weight < cfg.rewards[spec["discriminator"]].weight


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_penalty_weight_signs(key):
    spec = FAMILY[key]
    cfg = spec["factory"]()
    for name in BASE_MJLAB_COSTS:
        assert cfg.rewards[name].weight <= 0.0, (key, name)
    # torque rate is introduced at 0 and ramped negative only AFTER the skill
    # exists (standup timing lesson) — its final curriculum stage must be < 0.
    stages = cfg.curriculum["torque_rate_weight"].params["weight_stages"]
    assert stages[0]["weight"] == 0.0
    assert stages[-1]["weight"] < 0.0
    # self-negating microduck penalty → POSITIVE weight
    assert cfg.rewards["gentle_landing"].func is microduck_mdp.trunk_vertical_accel_penalty
    assert cfg.rewards["gentle_landing"].weight > 0.0


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_drift_taxes_per_task(key):
    spec = FAMILY[key]
    cfg = spec["factory"]()
    if spec["yaw_tol"] is None:
        # JumpHigh: verticality lives INSIDE the payment (mdp.JUMP_VERT_TOL_M
        # gate) — the small drift nudges were orders of magnitude too weak to
        # buy it (85.7% sideways leaps measured) and are dead weight here
        assert "jump_yaw_drift" not in cfg.rewards, key
        assert "jump_lateral_drift" not in cfg.rewards, key
        assert "jump_forward_drift" not in cfg.rewards, key
        return
    assert cfg.rewards["jump_yaw_drift"].weight == -0.2, key
    assert cfg.rewards["jump_yaw_drift"].params["yaw_tol_rad"] == spec["yaw_tol"], key
    assert cfg.rewards["jump_lateral_drift"].weight == -0.2, key
    assert cfg.rewards["jump_lateral_drift"].params["lat_tol_m"] == spec["lat_tol"], key
    # only the long-jump tasks may drift forward (it is their paid maneuver);
    # no family member registers the forward nudge anymore
    assert "jump_forward_drift" not in cfg.rewards, key


def test_task_constants_sane():
    assert JUMP_TARGET_X > 0.0
    assert JUMP_TARGET_Z > 0.0
    assert 0.0 < microduck_mdp.JUMP_MIN_FLIGHT_S < JUMP_AIR_TIME_CAP_S
    assert JUMP_APEX_CEILING > STAND_Z
    # the high task's launch potential must keep paying through a target-height
    # flight (0.115 + 0.05 apex) without unbounding the violent regime
    assert JUMP_HIGH_APEX_CEILING >= STAND_Z + JUMP_TARGET_Z
    assert JUMP_HIGH_APEX_CEILING < STAND_Z + 2.5 * JUMP_TARGET_Z
    assert microduck_mdp.JUMP_LAND_TILT_DEG > 0.0
    # the ankle gate is a WEAKENED copy of the trunk apex gate: it must never
    # exceed it, or it would gate out ballistic flights the trunk gate allows
    assert 0.0 < microduck_mdp.JUMP_MIN_FOOT_GAIN < microduck_mdp.JUMP_MIN_APEX_GAIN
    # the stance gate sits between a trot contact (~0.1-0.2 s) and a real
    # frog settle (~0.4 s+); short enough to keep a brisk frog cadence
    assert 0.2 <= microduck_mdp.JUMP_MIN_STANCE_S <= 1.0


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_protection_ramp_reaches_full_weight(key):
    cfg = FAMILY[key]["factory"]()
    ramped = (
        "gentle_landing",
        "head_impact",
        "trunk_impact",
        "servo_acc_spike",
        "servo_stall",
    )
    for name in ramped:
        stages = cfg.curriculum[f"{name}_weight"].params["weight_stages"]
        assert stages[0]["step"] == 0
        assert math.isclose(stages[0]["weight"], cfg.rewards[name].weight)
        assert stages[-1]["step"] == jump_cfg_mod.PROTECT_FULL_STEP
        assert math.copysign(1, stages[-1]["weight"]) == math.copysign(1, stages[0]["weight"])
        assert abs(stages[-1]["weight"]) > abs(stages[0]["weight"])


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_smoothness_curricula_stages_start_at_cfg_weight(key):
    cfg = FAMILY[key]["factory"]()
    for term in ("action_rate_weight", "torque_rate_weight"):
        stages = cfg.curriculum[term].params["weight_stages"]
        assert stages[0]["step"] == 0
        reward = cfg.rewards[term.replace("_weight", "_l2") if term == "action_rate_weight" else "joint_torque_rate_l2"]
        assert stages[0]["weight"] == reward.weight
        assert stages[-1]["weight"] < stages[0]["weight"]


def test_jump_high_redirect_economics():
    # 2026-10-09 warm-start experiment: at the frog's consolidation values
    # (action_rate −0.1 from step 0, entropy 0.003) the best hopping episode
    # netted −0.21 (action_rate −0.28 alone vs +0.23 hop revenue) and the
    # warm-started run converged to standing still. The redirect task must
    # explore: smoothness floor at the exploration value, family ceiling
    # late, and the pre-consolidation entropy.
    cfg = make_microduck_jump_high_env_cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0] == {"step": 0, "weight": -0.02}
    assert stages[-1]["weight"] == -0.1
    assert cfg.rewards["action_rate_l2"].weight == -0.02
    assert MicroduckJumpHighRlCfg.algorithm.entropy_coef == 0.01
    # the consolidation tasks keep the frog's proven 0.003
    assert MicroduckFrogJumpRlCfg.algorithm.entropy_coef == 0.003
    assert MicroduckLongJumpRlCfg.algorithm.entropy_coef == 0.003


def test_vertical_tolerance_curriculum_wide_then_tight():
    # 2026-10-10: a FIXED 2 cm verticality gate had zero partial gradient on
    # the warm-started policy (its hops move 4-9 cm → every jump paid ≈0 →
    # standing again, it571). The gate must open wide enough to score the
    # current behavior, then close onto the production 2 cm bar. Only the
    # JumpHigh task carries it.
    cfg = make_microduck_jump_high_env_cfg()
    stages = cfg.curriculum["jump_vertical_tolerance"].params["tol_stages"]
    steps = [s["step"] for s in stages]
    assert steps == sorted(steps) and steps[0] == 0
    assert stages[0]["tol"] >= 0.08   # wide: the frog's hops must score
    assert stages[-1]["tol"] == microduck_mdp.JUMP_VERT_TOL_FINAL_M == 0.02
    assert all(stages[i]["tol"] > stages[i + 1]["tol"] for i in range(len(stages) - 1))
    for other in ("frog", "long"):
        assert "jump_vertical_tolerance" not in FAMILY[other]["factory"]().curriculum
    # the curriculum func really advances the module state the state machine
    # reads (fake env, no sim; stages apply with strict >, so push the
    # counter past the second stage's step)
    env = _fake_env()
    saved = microduck_mdp.jump_vertical_tolerance
    try:
        env.common_step_counter = 150
        val = float(microduck_mdp.jump_vertical_tolerance_curriculum(
            env, torch.zeros(0, dtype=torch.long),
            [{"step": 0, "tol": 0.09}, {"step": 100, "tol": 0.04}],
        )[0])
        # module state is a Python float through a float32 tensor round-trip
        assert val == pytest.approx(0.04)
        assert microduck_mdp.jump_vertical_tolerance == pytest.approx(0.04)
    finally:
        microduck_mdp.jump_vertical_tolerance = saved


def test_warm_start_follows_env_var():
    # the collapse must never silently re-shape fresh runs' curricula; it is
    # gated on the SAME switch as the mdp Patch-5 counter restart, so one
    # env var controls the whole warm start
    import importlib

    import mjlab_microduck.tasks.microduck_jump_env_cfg as m

    old = dict(os.environ)
    try:
        os.environ.pop("MICRODUCK_WARM_START", None)
        importlib.reload(m)
        assert m.WARM_START is False
        os.environ["MICRODUCK_WARM_START"] = "1"
        importlib.reload(m)
        assert m.WARM_START is True
    finally:
        os.environ.clear()
        os.environ.update(old)
        importlib.reload(m)


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_upright_is_deleted_like_roulade(key):
    # Run-1 lesson: the always-on upright paid ~0.5/step for simply standing —
    # a guaranteed salary that dominated the risky annuity and taught the
    # policy to stand still for 4000 iterations. It must stay deleted.
    cfg = FAMILY[key]["factory"]()
    assert "upright" not in cfg.rewards


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_landing_annuity_rate_refs_per_task(key):
    spec = FAMILY[key]
    cfg = spec["factory"]()
    dx_ref, gain_ref = (
        (JUMP_TARGET_X, 0.0) if spec["discriminator"] == "jump_distance"
        else (0.0, JUMP_TARGET_Z)
    )
    # Run-1/2 lessons: the rate must be maneuver-quality over the FULL target
    # (a flat rate is a standing jackpot; a half-target rate let a 2.9 cm
    # micro-hop collect the whole pot). JumpHigh's maneuver is HEIGHT — its
    # hops move ~nowhere, so an arc rate would pay a perfect vertical jump
    # nothing; it rates by the apex rise instead.
    for name in ("jump_landing", "jump_landing_sharp", "jump_stand_tax"):
        params = cfg.rewards[name].params
        assert params["dx_ref"] == dx_ref, (key, name)
        assert params["gain_ref"] == gain_ref, (key, name)
        assert "air_ref" not in params, (key, name)


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_launch_ceiling_per_task(key):
    spec = FAMILY[key]
    cfg = spec["factory"]()
    assert cfg.rewards["jump_launch"].params["ceiling"] == spec["apex_ceiling"]


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_episode_length_per_task(key):
    cfg = FAMILY[key]["factory"]()
    assert cfg.episode_length_s == FAMILY[key]["episode_length_s"]


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_reset_event_wiring(key):
    spec = FAMILY[key]
    cfg = spec["factory"]()
    keys = list(cfg.events.keys())
    assert "set_jump_state" in keys
    assert keys.index("set_jump_state") > keys.index("reset_robot_joints")
    # the mode flag IS the task economics (see the mdp.py jump section header)
    assert cfg.events["set_jump_state"].params["multi_flight"] is spec["multi_flight"]


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_terminations_keep_nan_guard_and_tilt_only_recycle(key):
    cfg = FAMILY[key]["factory"]()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations
    recycle = cfg.terminations["fallen_recycle"]
    assert recycle.func is microduck_mdp.fallen_too_long
    # gate_z 0.0 → tilt-ONLY: a deep countermovement crouch is legitimate and
    # must never count as fallen (z-only gates recycle mid-crouch).
    assert recycle.params["gate_z_below"] == 0.0
    assert recycle.params["gate_tilt_above_deg"] > 0.0
    assert 0.0 < recycle.params["max_duration_s"] < 2.0


def test_feet_sensor_matches_state_machine_and_model():
    cfg = make_microduck_frog_jump_env_cfg()
    names = [s.name for s in cfg.scene.sensors]
    assert microduck_mdp.JUMP_FEET_SENSOR in names
    feet = next(s for s in cfg.scene.sensors if s.name == microduck_mdp.JUMP_FEET_SENSOR)
    # the pattern must resolve against the actual standup (groundcontact) model
    import re

    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_standup_spec

    model = get_standup_spec().compile()
    geoms = {
        name
        for i in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)) is not None
    }
    matched = [g for g in geoms if re.fullmatch(feet.primary.pattern, g)]
    assert set(matched) == {"left_foot_collision", "right_foot_collision"}


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_symmetry_enabled_like_roulade(key):
    # sagittal maneuvers → mirror loss on (roulade precedent)
    assert FAMILY[key]["rl_cfg"].algorithm.symmetry_cfg is not None


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_experiment_names_are_disjoint(key):
    # FrogJump intentionally KEEPS "microduck_jump" (trained-run continuity);
    # the two new tasks must never write into it or each other.
    names = [FAMILY[k]["rl_cfg"].experiment_name for k in FAMILY_IDS]
    mine = FAMILY[key]["rl_cfg"].experiment_name
    assert names.count(mine) == 1
    if key != "frog":
        assert mine != MicroduckFrogJumpRlCfg.experiment_name


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_actor_observation_keeps_the_61d_slot_layout(key):
    cfg = FAMILY[key]["factory"]()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    for padded in ("head_command", "body_command"):
        assert padded in terms
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_obs_parity_within_family_and_roulade(key):
    # ACTOR parity is the deployment contract (ONNX slot layout): every jump
    # task must match roulade AND its siblings exactly, so checkpoints stay
    # hot-swappable across the family and the runtime assembly works
    # unchanged.
    from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
        make_microduck_roulade_env_cfg,
    )

    cfg = FAMILY[key]["factory"]()
    roulade = make_microduck_roulade_env_cfg()
    for grp in ("actor",):
        assert list(cfg.observations[grp].terms.keys()) == list(
            roulade.observations[grp].terms.keys()
        ), f"{key} actor obs layout diverges from roulade on group {grp}"
    for other_key in FAMILY_IDS:
        other = FAMILY[other_key]["factory"]()
        for grp in ("actor", "critic"):
            assert list(cfg.observations[grp].terms.keys()) == list(
                other.observations[grp].terms.keys()
            ), f"{key} vs {other_key} obs layout diverges on group {grp}"


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_critic_gets_height_state_actor_does_not(key):
    # The jump rewards are height-heavy (landing height, apex rise, stand-tax
    # shortfall) — the value function sees [trunk_z, ankle_zs] as privileged
    # information, exactly like velocity's critic gets foot_height via its
    # ray sensor. The ACTOR must NOT: the real robot has no height sensor
    # (sim2real) and the 61D actor contract is what deployment hot-swaps.
    cfg = FAMILY[key]["factory"]()
    assert "jump_height_state" in cfg.observations["critic"].terms, key
    assert "jump_height_state" not in cfg.observations["actor"].terms, key
    assert cfg.observations["critic"].terms["jump_height_state"].func is (
        microduck_mdp.jump_height_state
    )


@pytest.mark.parametrize("key", FAMILY_IDS)
def test_command_is_tiny_noise_parity_slot(key):
    cfg = FAMILY[key]["factory"]()
    cmd = cfg.commands["twist"]
    lo, hi = cmd.ranges.lin_vel_x
    assert abs(lo) <= 0.01 and abs(hi) <= 0.01


# ── state-machine unit tests (fake env, pure torch) ──────────────────────────

class _FakeSensor:
    def __init__(self, found):
        self.data = type("D", (), {"found": found})()


class _FakeScene:
    def __init__(self, sensors, origins, robot):
        self.sensors = sensors
        self.terrain = type("T", (), {"env_origins": origins})()
        self._robot = robot

    def __getitem__(self, name):
        assert name == "robot"
        return self._robot


class _FakeAssetData:
    """body 0 = trunk (unused by the state machine), 1/2 = ankles. The ankles
    ride with the root unless ``env.foot_pos`` pins them (planted-feet tests)."""

    def __init__(self, env):
        self._env = env

    @property
    def root_link_pos_w(self):
        return self._env.root_pos

    @property
    def root_link_quat_w(self):
        return self._env.root_quat

    @property
    def body_link_pos_w(self):
        env = self._env
        pos = torch.zeros(env.num_envs, 3, 3)
        pos[:, 0] = env.root_pos
        feet = env.foot_pos if env.foot_pos is not None else env.root_pos
        pos[:, 1] = feet
        pos[:, 2] = feet
        return pos


class _FakeAsset:
    def __init__(self, env):
        self.data = _FakeAssetData(env)

    def find_bodies(self, name_keys, preserve_order=False):
        assert tuple(name_keys) == ("ankle_left", "ankle_right")
        return [1, 2], ["ankle_left", "ankle_right"]


class _FakeEnv:
    """Duck-typed env for _update_jump_state: 2 envs, upright, feet down."""

    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = 0.02
        self.common_step_counter = 0
        self.root_pos = torch.zeros(num_envs, 3)
        self.root_pos[:, 2] = 0.115
        self.root_quat = torch.zeros(num_envs, 4)
        self.root_quat[:, 0] = 1.0
        self.foot_pos = None  # None → ankles follow the root (real flight)
        self.asset = _FakeAsset(self)
        self.scene = _FakeScene(
            {"feet_ground_contact": _FakeSensor(torch.ones(num_envs, 1))},
            torch.zeros(num_envs, 3),
            self.asset,
        )

    @property
    def found(self):
        return self.scene.sensors[microduck_mdp.JUMP_FEET_SENSOR].data.found

    def step(self):
        self.common_step_counter += 1


def _fake_env(multi: bool = True) -> _FakeEnv:
    """Fresh fake env with the state machine initialized and the task
    economics set the way reset_jump_state would (multi_flight flag)."""
    env = _FakeEnv()
    microduck_mdp._jump_state(env)
    env._jump_multi[:] = multi
    return env


def test_standing_never_latches():
    env = _fake_env()
    for _ in range(20):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    assert not bool(env._jump_flight_ok.any())
    assert not bool(env._jump_landed.any())
    assert float(env._jump_air_s.max()) == 0.0


def test_short_contact_breaks_never_latch():
    # a shuffle alternates contact every step: 0.02 s breaks < min flight
    env = _fake_env()
    for i in range(40):
        env.step()
        env.found[:] = 1.0 if i % 2 == 0 else 0.0
        microduck_mdp._update_jump_state(env, env.asset)
    assert not bool(env._jump_flight_ok.any())
    assert not bool(env._jump_landed.any())


def test_flight_latch_requires_apex_gain():
    # A no-contact stretch with a LEVEL trunk is a standing dropout (the run-1
    # exploit), not a flight — no latch, no annuity, no air-time payment.
    # The fake foot reference is primed the way reset_jump_state leaves it
    # after the first grounded frame (real ankle z at takeoff reference).
    env = _fake_env()
    env._jump_grounded_z[:] = 0.115  # takeoff z from reset_jump_state
    env._jump_foot_grounded_z[:] = 0.115
    for _ in range(8):
        env.step()
        env.found[:] = 0.0  # feet off, trunk level
        microduck_mdp._update_jump_state(env, env.asset)
        paid = float(microduck_mdp.jump_air_time(env).sum())
    assert not bool(env._jump_flight_ok.any())
    assert not bool(env._jump_landed.any())
    assert paid == 0.0
    # ...and a stretch that RISES is a real flight: same duration, z gain.
    env2 = _fake_env()
    env2._jump_grounded_z[:] = 0.115
    env2._jump_foot_grounded_z[:] = 0.115
    for _ in range(8):
        env2.step()
        env2.found[:] = 0.0
        env2.root_pos[:, 2] += 0.003  # 6 frames × 3 mm = 18 mm ≥ apex gate
        microduck_mdp._update_jump_state(env2, env2.asset)
    assert bool(env2._jump_flight_ok.all())
    assert float(microduck_mdp.jump_air_time(env2).sum()) > 0.0


def test_sway_dropout_with_planted_feet_never_latches():
    # Run-2 exploit hardening: the feet sensor drops out while standing and
    # the trunk sways up ≥8 mm relative to the single-frame local z — the
    # trunk apex gate ALONE still passes. The ankle gate closes it: planted
    # ankles do not move, so this is not flight — no latch, no air-time pay.
    env = _fake_env()
    env._jump_grounded_z[:] = 0.115
    env._jump_foot_grounded_z[:] = 0.115
    env.foot_pos = env.root_pos.clone()  # pin the ankles at grounded height
    for _ in range(8):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 2] += 0.003  # trunk sway: 6 × 3 mm = 18 mm apex gain
        microduck_mdp._update_jump_state(env, env.asset)
        paid = float(microduck_mdp.jump_air_time(env).sum())
    assert not bool(env._jump_flight_ok.any())
    assert not bool(env._jump_landed.any())
    assert paid == 0.0


def test_flight_latch_air_time_and_distance_paid_once():
    env = _fake_env(multi=True)
    target = 0.08
    paid = torch.zeros(env.num_envs)
    air_paid = torch.zeros(env.num_envs)
    # a real frog stance before the leap (0.4 s ≥ JUMP_MIN_STANCE_S)
    for _ in range(20):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    # 6 airborne steps advancing 0.0125 m/step and rising 3 mm/step →
    # 0.075 m flight arc, 18 mm apex gain (clears the dropout gate)
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.0125
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
        air_paid += microduck_mdp.jump_air_time(env, air_time_cap_s=JUMP_AIR_TIME_CAP_S)
        paid += microduck_mdp.jump_distance_bonus(env, target_x=target)
    assert bool(env._jump_flight_ok.all())
    assert not bool(env._jump_landed.any())
    # air time pays during flight, uncapped here (0.12 s < cap)
    assert torch.allclose(air_paid, torch.full((env.num_envs,), 0.12))
    # no distance before the landing latch
    assert torch.all(paid == 0.0)
    # touchdown → latch fires, the flight arc is paid once
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    paid += microduck_mdp.jump_distance_bonus(env, target_x=target)
    assert bool(env._jump_landed.all())
    assert torch.allclose(paid, torch.full((env.num_envs,), 0.075 / target))
    # holding the landing pays no further distance (no jackpot)
    for _ in range(5):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
        paid += microduck_mdp.jump_distance_bonus(env, target_x=target)
    assert torch.allclose(paid, torch.full((env.num_envs,), 0.075 / target))


def test_distance_bonus_caps_at_target():
    env = _fake_env(multi=True)
    target = 0.05
    for _ in range(20):  # frog stance: 0.4 s ≥ JUMP_MIN_STANCE_S
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    # long flight, 0.10 m arc > target, rising 3 mm/step
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.0167
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    paid = float(microduck_mdp.jump_distance_bonus(env, target_x=target)[0])
    assert paid == pytest.approx(1.0)  # capped: overshooting forfeits the excess


def test_walking_between_hops_earns_no_distance():
    env = _fake_env(multi=True)
    target = 0.08
    # walk forward 0.10 m while grounded (20 steps, 0.4 s stance)
    for _ in range(20):
        env.step()
        env.root_pos[:, 0] += 0.005
        microduck_mdp._update_jump_state(env, env.asset)
    # vertical hop: 6 airborne steps rising 3 mm/step, NO forward motion
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    assert bool(env._jump_flight_ok.all())
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    paid = float(microduck_mdp.jump_distance_bonus(env, target_x=target)[0])
    # pays the hop's own arc (~0), NOT the walked distance (0.05/target)
    assert paid == pytest.approx(0.0, abs=1e-6)
    # and the arc-only annuity stays closed: a vertical hop has arc ≈ 0
    assert float(microduck_mdp._jump_annuity_rate(env, dx_ref=0.05).max()) == 0.0


def test_height_bonus_pays_the_flight_apex_rise():
    # THE JumpHigh discriminator: a qualified flight pays its own apex rise
    # above the takeoff z, capped at the target — NOT the crouch depth, NOT
    # walked distance.
    env = _fake_env(multi=False)
    target = 0.05
    for _ in range(20):  # frog stance: 0.4 s ≥ JUMP_MIN_STANCE_S
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    # vertical hop: 6 airborne steps rising 3 mm/step → 18 mm apex rise
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    paid = float(microduck_mdp.jump_height_bonus(env, target_z=target)[0])
    assert paid == pytest.approx(0.018 / target)
    # the gain-based annuity rate follows the same rise
    rate = float(microduck_mdp._jump_annuity_rate(env, dx_ref=0.0, gain_ref=target)[0])
    assert rate == pytest.approx(0.018 / target)


def test_height_bonus_caps_at_target_and_ignores_walking():
    env = _fake_env(multi=False)
    target = 0.05
    # walk forward 0.10 m while grounded, then a big vertical hop:
    # 18 airborne steps × 3 mm = 54 mm rise > target
    for _ in range(20):
        env.step()
        env.root_pos[:, 0] += 0.005
        microduck_mdp._update_jump_state(env, env.asset)
    for _ in range(18):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    # capped: overshooting forfeits the excess (no violence jackpot)...
    assert float(microduck_mdp.jump_height_bonus(env, target_z=target)[0]) == pytest.approx(1.0)
    # ...and walked distance never leaks into the height payment
    dist = float(microduck_mdp.jump_distance_bonus(env, target_x=0.08)[0])
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_height_bonus_vertical_gate_pays_in_place_only():
    # THE task definition (user, 2026-10-10): vertical means VERTICAL. The
    # same rise paid from a takeoff-point displacement collapses through the
    # verticality Gaussian: in place = full, tol off ≈ 1/e, 5×tol ≈ 0
    # (the shipped policy leapt 5.2 cm sideways — it must earn ~nothing).
    # The tolerance is curriculum module state; pin it explicitly and restore.
    tol = 0.02
    saved_tol = microduck_mdp.jump_vertical_tolerance
    microduck_mdp.jump_vertical_tolerance = tol
    try:
        _assert_vertical_gate(tol)
    finally:
        microduck_mdp.jump_vertical_tolerance = saved_tol


def _assert_vertical_gate(tol: float) -> None:
    def run_hop(dx_total: float) -> tuple[float, float]:
        env = _fake_env(multi=False)
        target = 0.05
        for _ in range(20):  # frog stance before the leap
            env.step()
            microduck_mdp._update_jump_state(env, env.asset)
        steps = 8
        for _ in range(steps):  # 8 airborne steps: 24 mm rise, spread drift
            env.step()
            env.found[:] = 0.0
            env.root_pos[:, 0] += dx_total / steps
            env.root_pos[:, 2] += 0.003
            microduck_mdp._update_jump_state(env, env.asset)
        env.step()
        env.found[:] = 1.0
        microduck_mdp._update_jump_state(env, env.asset)
        bonus = float(microduck_mdp.jump_height_bonus(env, target_z=target)[0])
        rate = float(microduck_mdp._jump_annuity_rate(env, dx_ref=0.0, gain_ref=target)[0])
        return bonus, rate

    import math as _math

    full_bonus, full_rate = run_hop(0.0)
    # (rel=1e-4: the fake env accumulates z in float32 — 24 mm carries ~5e-7
    # absolute rounding, just past approx's default 1e-6 relative window)
    assert full_bonus == pytest.approx(0.024 / 0.05, rel=1e-4)
    assert full_rate == pytest.approx(0.024 / 0.05, rel=1e-4)
    # takeoff xy is frozen on the FIRST airborne frame (after the first
    # increment), so the touchdown displacement is dx × (steps−1)/steps
    disp_frac = 7.0 / 8.0
    diag_tol, rate_tol = run_hop(tol / disp_frac)  # lands exactly tol off
    assert diag_tol == pytest.approx(full_bonus * _math.exp(-1.0), rel=1e-3)
    assert rate_tol == pytest.approx(full_rate * _math.exp(-1.0), rel=1e-3)
    diag_far, rate_far = run_hop(5 * tol / disp_frac)  # lands 5×tol off
    assert diag_far < 0.02 * full_bonus  # ≈ 0: a leap buys nothing
    assert rate_far < 0.02 * full_rate



def test_fallen_flailing_never_latches():
    # on the back (90° tilt about y): both feet off the ground is NOT flight
    env = _fake_env()
    env.root_quat[:, 0] = math.cos(math.pi / 4)
    env.root_quat[:, 2] = math.sin(math.pi / 4)
    for _ in range(30):
        env.step()
        env.found[:] = 0.0
        microduck_mdp._update_jump_state(env, env.asset)
    assert not bool(env._jump_flight_ok.any())
    assert float(microduck_mdp.jump_air_time(env).sum()) == 0.0
    assert not bool(env._jump_landed.any())


def test_tilted_landing_never_opens_the_annuity():
    # flight executed entirely past the tilt gate: no latch, no distance
    env = _fake_env()
    env.root_quat[:, 0] = math.cos(math.pi / 6)  # 60° tilt
    env.root_quat[:, 2] = math.sin(math.pi / 6)
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.0125
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    assert not bool(env._jump_landed.any())
    assert float(microduck_mdp.jump_distance_bonus(env, target_x=0.08)[0]) == 0.0


def test_air_time_cap_bounds_hop_chains():
    env = _fake_env()
    # one long flight (0.4 s), rising 3 mm/step: pays exactly the cap, not
    # the full duration
    total = 0.0
    for _ in range(20):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
        total += float(microduck_mdp.jump_air_time(env, air_time_cap_s=JUMP_AIR_TIME_CAP_S).sum())
    # 2 fake envs × cap: a 0.4 s flight pays 0.35 s worth, not the full 0.4 s
    assert total == pytest.approx(2 * JUMP_AIR_TIME_CAP_S)


def test_second_bigger_jump_updates_the_frontier():
    # Multi mode (FrogJump), run-4 per-flight semantics: EVERY qualified
    # touchdown pays its own arc — a second, bigger jump in the same episode
    # pays again (the run-3 frontier form paid only the episode improvement,
    # so chaining bought nothing and "jump once, then freeze" was optimal).
    env = _fake_env(multi=True)
    target = 0.08
    for _ in range(20):  # frog stance before leap 1
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    # flight 1: 4 airborne steps × 5 mm forward, rising 3 mm/step → 2 cm arc
    for _ in range(4):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.005
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    paid = float(microduck_mdp.jump_distance_bonus(env, target_x=target)[0])
    assert paid == pytest.approx(0.02 / target)
    assert bool(env._jump_landed.all())
    for _ in range(20):  # frog stance before leap 2 (0.4 s ≥ gate)
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    # flight 2 (same episode): 6 airborne steps × 12.5 mm → 7.5 cm arc
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.0125
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    paid += float(microduck_mdp.jump_distance_bonus(env, target_x=target)[0])
    # per-flight payment: 2 cm + 7.5 cm, each capped at the target
    assert paid == pytest.approx((0.02 + 0.075) / target)
    # the annuity rate follows the LAST flight, not the episode frontier
    rate = float(microduck_mdp._jump_annuity_rate(env, dx_ref=0.05)[0])
    assert rate == pytest.approx(min(0.075, 0.05) / 0.05)


def test_landing_window_rearms_per_flight_and_times_out():
    # Multi mode (FrogJump), run-4: the annuity is a WINDOW — open after each
    # touchdown, closed by the next qualified flight or after
    # JUMP_LAND_WINDOW_S of standing. "Jump once, then freeze" must stop
    # paying (the user-visible bug that created the continuous task).
    env = _fake_env(multi=True)
    for _ in range(20):  # frog stance before leap 1
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    # flight 1: 4 airborne steps × 5 mm, rising 3 mm/step → qualifies
    for _ in range(4):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.005
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    assert bool(env._jump_landed.all())
    # standing 0.8 s (< JUMP_LAND_WINDOW_S): the window stays open...
    for _ in range(40):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    assert bool(env._jump_landed.all())
    # ...but 1.6 s of standing (> window) closes it: freeze collects nothing
    for _ in range(40):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    assert not bool(env._jump_landed.any())
    # a second flight re-arms the qualification mid-air and the touchdown
    # reopens the window with the NEW flight's arc
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.0125
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    assert not bool(env._jump_landed.any())  # window closed during flight
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    assert bool(env._jump_landed.all())
    assert float(microduck_mdp.jump_distance_bonus(env, target_x=0.08)[0]) == pytest.approx(
        0.075 / 0.08
    )


def test_trot_cycles_never_pay():
    # Run-5 (user-visible): a bipedal RUN is a chain of short ballistic hops
    # that passes every flight gate — only the RHYTHM separates it from a
    # frog jump. A touchdown from a <0.35 s stance fires no payment and
    # opens no window; after a real frog stance the same flight pays.
    env = _fake_env(multi=True)

    def hop(steps_air, dx_per_step, dz_per_step=0.003):
        for _ in range(steps_air):
            env.step()
            env.found[:] = 0.0
            env.root_pos[:, 0] += dx_per_step
            env.root_pos[:, 2] += dz_per_step
            microduck_mdp._update_jump_state(env, env.asset)
        env.step()
        env.found[:] = 1.0
        microduck_mdp._update_jump_state(env, env.asset)

    # TROT: 8 grounded steps (0.16 s < JUMP_MIN_STANCE_S) then a hop
    for _ in range(8):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    hop(6, 0.0104)  # ~6 cm arc — would pay in full if it were a jump
    assert not bool(env._jump_touchdown.any())
    assert not bool(env._jump_landed.any())
    assert float(microduck_mdp.jump_distance_bonus(env, target_x=0.08).max()) == 0.0
    # more trotting never accumulates into pay
    for _ in range(3):
        for _ in range(8):
            env.step()
            microduck_mdp._update_jump_state(env, env.asset)
        hop(6, 0.0104)
    assert float(microduck_mdp.jump_distance_bonus(env, target_x=0.08).max()) == 0.0
    # FROG: 20 grounded steps (0.4 s) then the same hop — pays
    for _ in range(20):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    hop(6, 0.0104)
    assert bool(env._jump_touchdown.all())
    assert bool(env._jump_landed.all())
    assert float(microduck_mdp.jump_distance_bonus(env, target_x=0.08).max()) == pytest.approx(
        0.0624 / 0.08
    )


def test_single_mode_pays_only_the_first_jump():
    # Single mode (LongJump / JumpHigh): exactly ONE gated touchdown pays per
    # episode. A second, bigger jump pays nothing — the one-maneuver tasks
    # price exactly one jump (the pre-run-4 economics, correct for THESE
    # tasks: "jump once, then stand" is the definition, not the run-1-3 bug).
    env = _fake_env(multi=False)
    target = 0.08
    for _ in range(20):  # frog stance before leap 1
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    # flight 1: 4 airborne steps × 5 mm forward, rising 3 mm/step → 2 cm arc
    for _ in range(4):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.005
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    paid = float(microduck_mdp.jump_distance_bonus(env, target_x=target)[0])
    assert paid == pytest.approx(0.02 / target)
    assert bool(env._jump_landed.all())
    assert bool(env._jump_first_done.all())
    for _ in range(20):  # frog stance before leap 2
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    # flight 2 (bigger!): 6 airborne steps × 12.5 mm → 7.5 cm arc
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.0125
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    # no second payment, no second touchdown event...
    assert not bool(env._jump_touchdown.any())
    assert float(microduck_mdp.jump_distance_bonus(env, target_x=target)[0]) == 0.0
    paid += float(microduck_mdp.jump_distance_bonus(env, target_x=target)[0])
    assert paid == pytest.approx(0.02 / target)
    # ...and the rate basis stays LOCKED to the first jump
    rate = float(microduck_mdp._jump_annuity_rate(env, dx_ref=target)[0])
    assert rate == pytest.approx(0.02 / target)


def test_single_mode_window_never_times_out():
    # Single mode: after THE jump, standing until the timeout IS the task's
    # end state — the landing window must never close (unlike multi mode,
    # where freeze-after-one-jump was exactly the run-4 bug this window
    # semantics killed).
    env = _fake_env(multi=False)
    for _ in range(20):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    for _ in range(4):  # flight: 4 airborne steps × 5 mm, rising 3 mm/step
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.005
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    assert bool(env._jump_landed.all())
    # 1.6 s of standing — far past JUMP_LAND_WINDOW_S — keeps paying
    for _ in range(80):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    assert bool(env._jump_landed.all())
    rate = float(microduck_mdp._jump_annuity_rate(env, dx_ref=0.08)[0])
    assert rate == pytest.approx(0.02 / 0.08)


def test_single_mode_trot_hop_does_not_consume_the_jump():
    # A short-stance hop (trot rhythm) pays nothing in single mode and must
    # NOT consume the episode's one jump: the first FROG-rhythm touchdown
    # afterwards still pays.
    env = _fake_env(multi=False)
    target = 0.08
    # TROT: 8 grounded steps (0.16 s < gate) then a hop
    for _ in range(8):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.0104
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    assert not bool(env._jump_touchdown.any())
    assert float(microduck_mdp.jump_distance_bonus(env, target_x=target).max()) == 0.0
    assert not bool(env._jump_first_done.any())
    # real frog stance, then the jump that counts
    for _ in range(20):
        env.step()
        microduck_mdp._update_jump_state(env, env.asset)
    for _ in range(6):
        env.step()
        env.found[:] = 0.0
        env.root_pos[:, 0] += 0.0125
        env.root_pos[:, 2] += 0.003
        microduck_mdp._update_jump_state(env, env.asset)
    env.step()
    env.found[:] = 1.0
    microduck_mdp._update_jump_state(env, env.asset)
    assert float(microduck_mdp.jump_distance_bonus(env, target_x=target)[0]) == pytest.approx(
        0.075 / target
    )
