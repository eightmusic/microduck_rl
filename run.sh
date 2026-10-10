# cd /data/zhangshu/project/duck/microduck_rl

# WANDB_MODE=disabled uv run --no-sync train Mjlab-Jump-Transition-Flat-MicroDuck \
#   --env.scene.num-envs 4096 \
#   --agent.max-iterations 3000 \
#   --agent.load-run 2026-10-08_14-35-53_microduck_jump_transition \
#   --agent.load-checkpoint model_2398.pt \
#   --agent.resume True


# cd /data/zhangshu/project/duck/microduck_rl

# _VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-Jump-Transition-Flat-MicroDuck \
#   --checkpoint-file logs/rsl_rl/microduck_jump_transition/2026-10-08_14-35-53_microduck_jump_transition/model_2398.pt \
#   --no-terminations True #\
# #   > /tmp/jump_play.log 2>&1 < /dev/null &

# pkill -f "play Mjlab-Jump-Transition"

Mjlab-JumpHigh-Flat-MicroDuck


WANDB_MODE=disabled uv run --no-sync train Mjlab-JumpHigh-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000

WANDB_MODE=disabled uv run --no-sync train Mjlab-Spin-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000

WANDB_MODE=disabled uv run --no-sync train Mjlab-Jump-Transition-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000

_VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-Spin-Flat-MicroDuck  \
  --checkpoint-file logs/rsl_rl/spin/2026-10-08_16-14-10_spin/model_1000.pt

_VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-Spin-Flat-MicroDuck  \
  --checkpoint-file logs/rsl_rl/spin/2026-10-08_16-14-10_spin/model_0.pt

_VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-Jump-Transition-Flat-MicroDuck  \
  --checkpoint-file logs/rsl_rl/microduck_jump_transition/2026-10-08_19-03-07_microduck_jump_transition/model_2500.pt

_VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-Jump-Transition-Flat-MicroDuck  \
  --checkpoint-file logs/rsl_rl/microduck_jump_transition/2026-10-09_10-04-26_microduck_jump_transition/model_500.pt

_VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-Jump-Transition-Flat-MicroDuck  \
  --checkpoint-file logs/rsl_rl/microduck_jump/2026-10-09_11-04-05_microduck_jump/model_2499.pt

_VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-Jump-Transition-Flat-MicroDuck  \
  --checkpoint-file logs/rsl_rl/microduck_jump/2026-10-09_13-25-21_microduck_jump/model_1500.pt


uv run --no-sync play Mjlab-Jump-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_jump/2026-10-09_13-25-21_microduck_jump/model_2499.pt

_VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-Jump-Transition-Flat-MicroDuck  \
  --checkpoint-file logs/rsl_rl/microduck_jump/2026-10-09_13-25-21_microduck_jump/model_2499.pt

_VISER_PORT_OVERRIDE=8090 nohup uv run --no-sync play Mjlab-JumpHigh-Flat-MicroDuck  \
  --checkpoint-file logs/rsl_rl/microduck_jump_high/2026-10-09_16-54-02_microduck_jump_high/model_500.pt
