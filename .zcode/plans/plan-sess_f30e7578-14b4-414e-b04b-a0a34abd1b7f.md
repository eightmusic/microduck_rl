# 短程跳跃发现实验（保留正式 Jump 配方）

目标：在你启动大规模训练前，用 run-2 checkpoint 做一个低成本、可量化的发现阶段实验；先证明策略能产生真实起跳，再决定是否值得精训。不会覆盖已有 `Mjlab-Jump-Flat-MicroDuck` 配方或训练产物。

## 1. 新增隔离的 Discovery 变体

新增 `make_microduck_jump_discovery_env_cfg()` 与独立 task ID，例如 `Mjlab-Jump-Discover-Flat-MicroDuck`，沿用相同机器人、61D observation、DR 和 jump 状态机，因此可从现有 run-2 的 `model_3999.pt` warm-start。

Discovery 阶段暂时把目标限定为“离地并获得向上高度”，而不是跳远/站稳：
- 使用有界的 trunk-height 势能进展项（standing 为零基线，向上才付、停留不持续付），显著提高它相对于尝试成本的质量；不采用无上限的向上速度奖励，避免暴力解。
- 暂时把距离与 landing 年金设为 0；保留 apex-gain 门和真实滞空测量，禁止接触传感器闪断制造假起跳。
- discovery 前 300 iterations 将 action-rate、body-angular-velocity、冲击/servo 保护等 attempt tax 降至接近 0；保持 NaN guard 和 crash recycle。若出现真实起跳，再逐阶段恢复保护与落地项。
- 单独 `experiment_name`，checkpoint 与正式 Jump run 完全分开。

## 2. 测试与静态验证

为 discovery cfg 增加 CPU cfg tests：检查 obs 仍为 episodic 61D 布局、jump apex gate 保留、距离/落地项在 discovery 阶段关闭、起跳势能项有正权重、Discovery task 注册成功、curriculum 阶段值与符号正确。先运行 jump cfg 测试套件。

## 3. 小规模试跑

先做 25-iteration smoke test，检查能构建、无 NaN、所有 reward 可计算、checkpoint 正常保存。通过后用：
- 1024 environments；
- 最多 300 iterations；
- `WANDB_MODE=disabled`；
- 从 `logs/rsl_rl/microduck_jump/2026-09-29_16-39-53_microduck_jump/model_3999.pt` warm-start（保留策略与 normalizer，课程计数从 0 开始）。

这约为正式 4096 env × 4000 iters 训练量的 1.9%，适合先探路。若 100–150 iterations 完全没有正向高度进展或真实 flight，提前停止并报告，不继续花预算。

## 4. 评估与交付

用无头 eval 评估 discovery 最佳 checkpoint（每个 env 确保完整回合，分别统计真实飞行、apex gain、滞空时长、前向弧长、落地率），并检查 TensorBoard 中 discovery reward 的趋势。交付：实际 run/checkpoint 路径、短训曲线摘要、eval 数字、是否建议进入距离/落地精训，以及建议的具体权重/门槛。若 300 iterations 内没有真实起跳，不启动更大的训练；给出基于测量的下一次调整方向。