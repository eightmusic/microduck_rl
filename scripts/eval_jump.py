"""Headless eval of a trained jump policy — measure what it ACTUALLY does.

AGENTS.md: measure before theorizing. Rolls the trained policy on the jump
env and reports per-episode: both-feet flight segments (with apex gain to
separate real hops from fall artifacts), landing-latch state, forward arc,
and end state. Also flags SENSOR DROPOUT while standing — frames where no
foot reports contact although the trunk is upright and near standing height
(the suspected false-flight source).

Usage:
    uv run --no-sync python scripts/eval_jump.py \
        --checkpoint logs/rsl_rl/microduck_jump/<run>/model_3999.pt \
        [--task Mjlab-FrogJump-Flat-MicroDuck] [--episodes 64]

Tasks (the jump family): FrogJump = continuous hopping, arc stats matter;
JumpHigh = vertical jump, the APEX RISE stats are the headline; LongJump =
one jump then stand (read flights-per-episode ≈ 1 and the end-state share).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls, load_rl_cfg

from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    JUMP_TARGET_Z,
    make_microduck_frog_jump_env_cfg,
    make_microduck_jump_high_env_cfg,
    make_microduck_long_jump_env_cfg,
)

JUMP_TASKS = {
    "Mjlab-FrogJump-Flat-MicroDuck": make_microduck_frog_jump_env_cfg,
    "Mjlab-LongJump-Flat-MicroDuck": make_microduck_long_jump_env_cfg,
    "Mjlab-JumpHigh-Flat-MicroDuck": make_microduck_jump_high_env_cfg,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(JUMP_TASKS), default="Mjlab-FrogJump-Flat-MicroDuck")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=2, help="episodes per env")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = JUMP_TASKS[args.task](play=True)
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    B = env.num_envs
    dt = env.step_dt
    dev = env.device

    wrapped = RslRlVecEnvWrapper(env)
    from dataclasses import asdict

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(load_rl_cfg(args.task)), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy(device=env.device)

    robot = env.scene["robot"]
    origins = env.scene.terrain.env_origins
    T = int(env.max_episode_length_s / dt)  # one full episode (auto-reset at T)

    n_runs = max(args.episodes, 1)
    agg = dict(eps=0, latched=0, with_real_flight=0, best_air=0.0, best_arc=0.0,
               dropouts=0.0, eps_with_dropout=0, stood_end=0)
    per_ep_flight_counts = []
    real_flight_arcs = []   # forward arc of each REAL flight segment
    real_flight_apexes = []  # trunk-z rise of each REAL flight segment (JumpHigh headline)
    real_flight_lats = []   # lateral displacement of each REAL flight segment
    real_flight_yaws = []   # |yaw change| during each REAL flight segment (deg)

    for run in range(n_runs):
        obs, _ = wrapped.reset()

        grounded_log = torch.zeros(B, T, dtype=torch.bool, device=dev)
        z_log = torch.zeros(B, T, device=dev)
        x_log = torch.zeros(B, T, device=dev)
        y_log = torch.zeros(B, T, device=dev)
        yaw_log = torch.zeros(B, T, device=dev)
        fwd_x = torch.ones(B, device=dev)
        fwd_y = torch.zeros(B, device=dev)
        tilt_log = torch.zeros(B, T, device=dev)

        with torch.no_grad():
            for t in range(T):
                action = policy(obs)
                obs, _, _, _ = wrapped.step(action)
                pos = (robot.data.root_link_pos_w - origins)[:B]
                q = robot.data.root_link_quat_w[:B]
                if t == 0:
                    yaw = torch.atan2(
                        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                        1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                    )
                    fwd_x, fwd_y = torch.cos(yaw), torch.sin(yaw)
                found = env.scene.sensors["feet_ground_contact"].data.found
                grounded_log[:, t] = (found.view(env.num_envs, -1) > 0).any(dim=-1)[:B]
                cos_tilt = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
                tilt_log[:, t] = torch.rad2deg(torch.acos(cos_tilt.clamp(-1, 1)))
                z_log[:, t] = pos[:, 2]
                # project world xy onto the spawn forward axis and its perpendicular
                x_log[:, t] = pos[:, 0] * fwd_x + pos[:, 1] * fwd_y
                y_log[:, t] = -pos[:, 0] * fwd_y + pos[:, 1] * fwd_x
                yaw_log[:, t] = torch.atan2(
                    2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                )
                # auto_reset keeps episodes rolling; per-episode slicing is by
                # time (episode length == T), matching the timeout.

        # per-env episode analysis (episode == this whole run window)
        for g in range(B):
            grounded = grounded_log[g]
            agg["eps"] += 1
            air = 0.0
            seg_start = 0
            best_air = 0.0
            best_arc = 0.0
            real_flights = 0
            last_gx = float(x_log[g, 0])
            last_gy = float(y_log[g, 0])
            for t in range(T):
                if grounded[t]:
                    if air > 0:
                        seg_gain = float(z_log[g, seg_start:t].max()) - float(z_log[g, seg_start])
                        seg_arc = float(x_log[g, t]) - last_gx
                        seg_lat = float(y_log[g, t]) - last_gy
                        best_air = max(best_air, air)
                        best_arc = max(best_arc, seg_arc)
                        # a REAL flight: rises >5 mm (the latch's trunk bar is
                        # 4 mm — the trained policy leaps LOW, so 1 cm would
                        # hide most of its jumps), short enough to not be a
                        # fall (long tumbling stretches drift low and slide)
                        if seg_gain > 0.005 and air < 0.4:
                            real_flights += 1
                            real_flight_arcs.append(seg_arc)
                            real_flight_apexes.append(seg_gain)
                            real_flight_lats.append(seg_lat)
                            dyaw = yaw_log[g, t] - yaw_log[g, seg_start]
                            dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
                            real_flight_yaws.append(float(dyaw.abs()) * 180.0 / 3.14159265)
                        air = 0.0
                    last_gx = float(x_log[g, t])
                    last_gy = float(y_log[g, t])
                else:
                    if air == 0.0:
                        seg_start = t
                    air += dt
            per_ep_flight_counts.append(real_flights)
            if best_air >= 0.06:
                agg["latched"] += 1
            if real_flights > 0:
                agg["with_real_flight"] += 1
            agg["best_air"] = max(agg["best_air"], best_air)
            agg["best_arc"] = max(agg["best_arc"], best_arc)
            # sensor dropouts while standing: no foot contact but upright & tall
            standing = (tilt_log[g] < 25.0) & (z_log[g] > 0.09)
            dropout = (~grounded) & standing
            n_drop = int(dropout.sum())
            agg["dropouts"] += n_drop
            agg["eps_with_dropout"] += int(n_drop > 0)
            tail = tilt_log[g, -50:].mean() < 25.0 and z_log[g, -50:].mean() > 0.09
            agg["stood_end"] += int(bool(tail))

    env.close()

    eps = max(agg["eps"], 1)
    print(f"\n=== eval: {agg['eps']} episodes ({args.num_envs} envs × {n_runs}) ===")
    print(f"episodes with ≥1 latch-qualifying flight (≥{0.06:.2f}s): "
          f"{agg['latched']} ({100 * agg['latched'] / eps:.1f}%)")
    print(f"episodes with ≥1 REAL flight (>1cm apex gain): "
          f"{agg['with_real_flight']} ({100 * agg['with_real_flight'] / eps:.1f}%)")
    print(f"best flight duration: {agg['best_air']:.3f} s | best arc: {agg['best_arc']:.3f} m")
    print(f"standing sensor dropouts: {agg['dropouts']} frames in "
          f"{agg['eps_with_dropout']} episodes "
          f"({100 * agg['eps_with_dropout'] / eps:.1f}% of episodes)")
    print(f"stood upright at episode end: {agg['stood_end']} ({100 * agg['stood_end'] / eps:.1f}%)")
    if per_ep_flight_counts:
        import collections
        hist = collections.Counter(per_ep_flight_counts)
        print("real-flights-per-episode histogram:", dict(sorted(hist.items())))
    if real_flight_arcs:
        import numpy as np
        arcs = np.abs(np.array(real_flight_arcs))
        apexes = np.array(real_flight_apexes)
        lats = np.abs(np.array(real_flight_lats))
        yaws = np.array(real_flight_yaws)
        print(f"real-flight APEX RISE (n={len(apexes)}): median={np.median(apexes)*1000:.1f} mm, "
              f"p90={np.percentile(apexes, 90)*1000:.1f} mm, max={apexes.max()*1000:.1f} mm "
              f"(JumpHigh: target {JUMP_TARGET_Z*1000:.0f} mm)")
        print(f"real-flight arc (n={len(arcs)}): median={np.median(arcs):.3f} m, "
              f"p90={np.percentile(arcs, 90):.3f} m, max={arcs.max():.3f} m")
        for thr in (0.01, 0.02, 0.05):
            share = 100.0 * float((arcs >= thr).mean())
            print(f"  real flights with |arc| ≥ {thr*100:.0f} cm: {share:.1f}%")
        print(f"real-flight LATERAL drift: median={np.median(lats):.3f} m, "
              f"p90={np.percentile(lats, 90):.3f} m | "
              f"share with |lat| > |arc| (sideways-dominant): "
              f"{100.0 * float((lats > arcs).mean()):.1f}%")
        print(f"real-flight yaw change: median={np.median(yaws):.1f}°, "
              f"p90={np.percentile(yaws, 90):.1f}°")


if __name__ == "__main__":
    main()
