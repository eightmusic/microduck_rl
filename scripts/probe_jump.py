"""Physics probe for the jump task — measure achievable jumps BEFORE training.

AGENTS.md workflow step 2: verify physics assumptions in sim before training.

Lesson from probe v1: the velstand deep-crouch anchor doubles as the roulade
tuck — a forward-pitched leg-fold pose, NOT an upright squat. Holding it
open-loop (BAM kp is low) face-plants the robot, so a "hold crouch then
launch" profile never reaches a launch. v2 therefore measures the MINIMUM
physics question — can the leg servos produce enough impulse to leave the
ground at all? — with pulse profiles launched straight from standing HOME:

  • ankle pulse:  plantarflexion burst (toe-off push), no crouch
  • extend pulse: full-leg extension burst (hip fwd / knee back / ankle fwd)
  • cm-lite:      a short 8-step dip, then the extend burst

Each env runs its grid point until its FIRST termination (time-out, crash
recycle) or the round ends; metrics are flushed at that moment — falling
over AFTER a measured jump does not erase the measurement.

Usage:
    uv run --no-sync python scripts/probe_jump.py [--device cuda:0] [--rounds 3]

Output: a per-profile table + a suggested JUMP_TARGET_X (0.7 × best arc among
profiles that land upright at least half the time).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_frog_jump_env_cfg

# HOME leg joint angles (absolute), servo-indexed (2-4 left, 11-13 right).
_HOME_ABS = {
    2: 0.4579, 3: 0.0049, 4: -0.4530,
    11: -0.4579, 12: -0.0049, 13: 0.4530,
}


def _off(targets: dict) -> torch.Tensor:
    """Action offset (target − HOME) from {servo_idx: absolute angle}."""
    off = torch.zeros(14)
    for idx, angle in targets.items():
        off[idx] = angle - _HOME_ABS[idx]
    return off


# Servo indices (0-4 left leg, 9-13 right leg).
L_HIP, L_KNEE, L_ANK = 2, 3, 4
R_HIP, R_KNEE, R_ANK = 11, 12, 13


# ── Profile grid ──────────────────────────────────────────────────────────────
# Each entry: (name, [(phase_str, steps), ...]). Names carry the amplitude
# ("a=" pulse amplitude, "d=" dip depth); phase words select the per-env
# action vectors built in main(). All profiles launch from standing HOME.
def _grid():
    g = []
    # FAST countermovement: dip then burst. Timing note: the BAM actuator
    # carries a 3-6 control-lag delay (~60-120 ms), so sub-100 ms pulses never
    # reach the motor (v3 lesson: 2-step bursts produced zero impulse). Keep
    # the total dip→burst window short enough that the topple instability has
    # no time to develop (v1 lesson: held crouches collapse).
    for dip in (0.5, 0.8):
        for dsteps in (8, 12):
            for esteps in (6, 10):
                g.append((f"cmfast d={dip} dn={dsteps} en={esteps}",
                          [("home", 15), ("crouch", dsteps), ("extend", esteps),
                           ("home", 120)]))
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=3, help="episodes per grid point")
    args = parser.parse_args()

    cfg = make_microduck_frog_jump_env_cfg(play=True)
    cfg.scene.num_envs = 16
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    dt = env.step_dt
    dev = env.device

    grid = _grid()
    G = len(grid)
    assert G <= cfg.scene.num_envs

    # Per-env phase actions: (G, 14) for each named phase of each profile.
    phase_names = ["home", "ankle", "extend", "crouch"]
    phase_actions = torch.zeros(G, len(phase_names), 14, device=dev)
    for g, (name, plan) in enumerate(grid):
        amplitudes = {}
        for key, word in (("a=", "ankle"), ("a=", "extend"), ("d=", "cm")):
            pass
        # decode amplitudes from the profile name (single param per profile)
        for token in name.split():
            if "=" in token:
                key, val = token.split("=")
                amplitudes[key] = float(val)
        if name.startswith("ankle"):
            amp = amplitudes["a"]
            phase_actions[g, 1] = _off(
                {L_ANK: _HOME_ABS[L_ANK] - amp, R_ANK: _HOME_ABS[R_ANK] + amp}
            ).to(dev)
        elif name.startswith("cmfast"):
            dip = amplitudes["d"]
            phase_actions[g, 3] = _off({
                L_HIP: _HOME_ABS[L_HIP] - dip, L_KNEE: _HOME_ABS[L_KNEE] + dip,
                L_ANK: _HOME_ABS[L_ANK] + 0.5 * dip,
                R_HIP: _HOME_ABS[R_HIP] + dip, R_KNEE: _HOME_ABS[R_KNEE] - dip,
                R_ANK: _HOME_ABS[R_ANK] - 0.5 * dip,
            }).to(dev)
            burst = 1.5 * dip
            phase_actions[g, 2] = _off({
                L_HIP: _HOME_ABS[L_HIP] + burst, L_KNEE: _HOME_ABS[L_KNEE] - burst,
                L_ANK: _HOME_ABS[L_ANK] - burst,
                R_HIP: _HOME_ABS[R_HIP] - burst, R_KNEE: _HOME_ABS[R_KNEE] + burst,
                R_ANK: _HOME_ABS[R_ANK] + burst,
            }).to(dev)
        # "ankle" family: no longer in the grid (v3: zero flight even at the
        # joint limit) but the parser stays for reuse.

    # Phase index tensor per profile: (G, T)
    word_to_idx = {"home": 0, "ankle": 1, "extend": 2, "crouch": 3}
    T = max(sum(n for _, n in plan) for _, plan in grid)
    plan_t = torch.zeros(G, T, dtype=torch.long, device=dev)
    for g, (_, plan) in enumerate(grid):
        t = 0
        for word, steps in plan:
            plan_t[g, t:t + steps] = word_to_idx[word]
            t += steps

    def actions_at(t: int) -> torch.Tensor:
        idx = plan_t[:, t]  # (G,)
        act = phase_actions[torch.arange(G, device=dev), idx]  # (G, 14)
        if env.num_envs > G:  # spare envs hold HOME
            act = torch.cat(
                [act, torch.zeros(env.num_envs - G, 14, device=dev)], dim=0
            )
        return act

    stats = [dict(n=0, flights=0.0, best_air=0.0, best_arc=0.0, apex=0.0,
                  dipz=1e9, az=0.0, stood=0.0) for _ in range(G)]

    for round_idx in range(args.rounds):
        env.reset()
        robot = env.scene["robot"]
        origins = env.scene.terrain.env_origins

        grounded_log = torch.ones(G, T, dtype=torch.bool, device=dev)
        x_log = torch.zeros(G, T, device=dev)
        z_log = torch.zeros(G, T, device=dev)
        tilt_log = torch.zeros(G, T, device=dev)
        alive = torch.ones(G, dtype=torch.bool, device=dev)
        end_t = torch.full((G,), T - 1, dtype=torch.long, device=dev)
        fwd_x = torch.ones(G, device=dev)
        fwd_y = torch.zeros(G, device=dev)

        for t in range(T):
            _, _, terminated, truncated, _ = env.step(actions_at(t))
            pos = (robot.data.root_link_pos_w - origins)[:G]
            q = robot.data.root_link_quat_w[:G]
            if t == 0:
                yaw = torch.atan2(
                    2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                )
                fwd_x, fwd_y = torch.cos(yaw), torch.sin(yaw)
            cos_tilt = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
            found = env.scene.sensors["feet_ground_contact"].data.found
            grounded_log[:, t] = (found.view(env.num_envs, -1) > 0).any(dim=-1)[:G]
            x_log[:, t] = pos[:, 0] * fwd_x + pos[:, 1] * fwd_y
            z_log[:, t] = pos[:, 2]
            tilt_log[:, t] = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
            done = (((terminated + truncated) > 0)[:G]) & alive
            if bool(done.any()):
                end_t[done] = t
                alive &= ~done
            if not bool(alive.any()):
                break

        # flush metrics per env over [0, end_t]
        for g in range(G):
            n = int(end_t[g]) + 1
            grounded = grounded_log[g, :n]
            air = 0.0
            seg_start = 0.0
            best_air = 0.0
            best_arc = 0.0
            real_air = 0.0
            real_arc = 0.0
            n_real = 0
            last_gx = float(x_log[g, 0])
            for t in range(n):
                if grounded[t]:
                    if air > 0:
                        seg_gain = float(z_log[g, seg_start:t].max()) - float(z_log[g, seg_start])
                        arc = float(x_log[g, t]) - last_gx
                        best_air = max(best_air, air)
                        best_arc = max(best_arc, arc)
                        # a REAL ballistic flight gains >1 cm of height and is
                        # short; a fall artifact drifts low and can last long.
                        if seg_gain > 0.01 and air < 0.4:
                            n_real += 1
                            real_air = max(real_air, air)
                            real_arc = max(real_arc, arc)
                        air = 0.0
                    last_gx = float(x_log[g, t])
                else:
                    if air == 0.0:
                        seg_start = t
                    air += dt
            apex = float(z_log[g, :n].max())
            dipz = float(z_log[g, :n].min())  # lowest trunk z (dip depth)
            max_az = float(
                ((z_log[g, 1:n] - z_log[g, : n - 1]) / dt).abs().max()
            )
            tail = max(n - 10, 0)
            stood = (
                float(tilt_log[g, tail:n].mean()) < 20.0
                and float(z_log[g, tail:n].mean()) > 0.10
                and bool(grounded[tail:n].all())
            )
            s = stats[g]
            s["n"] += 1
            s["flights"] += n_real
            s["best_air"] = max(s["best_air"], real_air)
            s["best_arc"] = max(s["best_arc"], real_arc)
            s["apex"] = max(s["apex"], apex)
            s["dipz"] = min(s["dipz"], dipz)
            s["az"] = max(s["az"], max_az)
            s["stood"] += float(stood)

    env.close()

    print(f"\ndt={dt:.3f}s  profiles={G}  rounds={args.rounds}   "
          f"(flights/air/arc count REAL ballistic flights only: >1 cm apex gain, <0.4 s)")
    hdr = (f"{'profile':<20} {'eps':>3} {'rflights':>8} {'air_s':>6} "
           f"{'arc_m':>6} {'apex_m':>6} {'dip_m':>6} {'maxaz':>6} {'stood':>5}")
    print(hdr)
    print("-" * len(hdr))
    for (name, _), s in zip(grid, stats):
        n = max(s["n"], 1)
        print(f"{name:<20} {s['n']:3d} {s['flights'] / n:8.1f} {s['best_air']:6.3f} "
              f"{s['best_arc']:6.3f} {s['apex']:6.3f} {s['dipz']:6.3f} {s['az']:6.1f} {s['stood'] / n:5.2f}")

    candidates = [s["best_arc"] for s in stats
                  if s["n"] >= 1 and s["stood"] / max(s["n"], 1) >= 0.5
                  and s["best_arc"] > 0.02]
    if not candidates:
        print("\nNO profile produced a clean landing-then-standing jump. If any "
              "profile DID fly (air_s > 0) but never landed upright, flight is "
              "physically achievable but the landing is the hard part — still "
              "report before setting the target.")
        return
    best = max(candidates)
    print(f"\nBest clean arc: {best:.3f} m → suggested JUMP_TARGET_X ≈ "
          f"{0.7 * best:.3f} m (0.7 × best)")


if __name__ == "__main__":
    main()
