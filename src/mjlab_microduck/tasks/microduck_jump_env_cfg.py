"""Microduck 跳跃任务家族 —— 一套共享配方支撑的三个任务。

最初的单一"立定跳远"任务于 2026-10-09 拆分为：

  ┌──────────────────────┬───────────────────────────────┬───────────────┐
  │ 任务 ID              │ 动作                          │ 判别项        │
  ├──────────────────────┼───────────────────────────────┼───────────────┤
  │ Mjlab-JumpHigh-      │ 单次垂直向上跳，越高越好，    │ jump_height   │
  │   Flat-MicroDuck     │ 落地站稳                      │ （apex 增量） │
  │ Mjlab-LongJump-      │ 单次向前跳向目标弧长，        │ jump_distance │
  │   Flat-MicroDuck     │ 落地站稳                      │ （前向弧长）  │
  │ Mjlab-FrogJump-      │ 连续前向蛙跳，每一跳都付分    │ jump_distance │
  │   Flat-MicroDuck     │                               │ （逐次飞行）  │
  └──────────────────────┴───────────────────────────────┴───────────────┘

三个任务共享同一套机器人/观测/DR/锁存配方（_make_jump_base）和同一个逐
环境状态机（mdp.py 跳跃段），模式标志在 reset 时写入：
  • multi_flight=True（FrogJump）：每次合格触地都付分，站立
    JUMP_LAND_WINDOW_S 后落地窗口关闭 —— 连跳是好跳就是严格最优。
  • multi_flight=False（JumpHigh / LongJump）：每回合只付第一次合格触地，
    年金率锁定在那一次跳上，窗口一旦打开就永不关闭 —— "完成那一次动作
    然后站稳"就是这两个任务的目标终态，因此跳跃后的逐步落地年金是设计
    意图，而非 run-1-3 的漏洞（当时的问题在于平价年金 + 可被站立骗开的
    锁存，两者均已在状态机里修复）。

实验名：FrogJump 保留 `microduck_jump` —— run-4 生产训练
（logs/rsl_rl/microduck_jump/，如 2026-10-09_13-25-21 的 model_2499 及其
导出 ONNX）就是这个任务的已训练策略，不用改任何管线即可继续
resume/评估。两个新任务分别写入 `microduck_jump_high` / `microduck_long_jump`。

WARM-START 配方（新任务不要从头训 —— 跳跃技能已经存在；2026-10-09 一次
从头训的 1750-iter JumpHigh run 收敛到站立不动，正是 run-1 的失败形态：
尝试有成本、所有任务奖励都在飞行锁后面、站立 = 0 分）：
  • LongJump / JumpHigh 从已训练的 frog checkpoint 暖启动：
      MICRODUCK_WARM_START=1 uv run train Mjlab-LongJump-Flat-MicroDuck \
        --env.scene.num-envs 4096 \
        --agent.load-checkpoint <frog run>/model_2499.pt --agent.resume True
    MICRODUCK_WARM_START=1 是唯一的暖启动开关：mdp.py 的 Patch 5 重置步数
    计数器，同时环境变量控制的 WARM_START（下方）把所有课程从 step 0 钉在
    末段（AGENTS.md 暖启动规则 —— 加载的策略已经受训于满 DR/满保护条件；
    把保护从 25% 重新爬坡、平滑度从 0 重新放开，等于把源策略已经掌握的
    东西重新放松）。三个任务的 obs 布局、normalizer 家族和 DR 完全一致，
    normalizer 原样迁移。
  • critic 携带特权高度观测（jump_height_state，2026-10-09 新增），拆分前
    的 frog checkpoint 没有它：rsl_rl 的全量加载会在 critic 形状上崩溃。
    Patch 6（mdp.py）捕获该错误并只加载 ACTOR，critic + optimizer 重新
    初始化 —— 策略保住技能，价值函数几百 iter 内重学。跨过该边界的 run
    会在启动时打印一次 "[mdp] Patch 6: checkpoint CRITIC obs mismatch"。
  ⚠️ 薪资悬崖警告（run-4，2026-10-09）：永远不要把生产训练过的跳跃
    checkpoint 暖启动进带薪资阶段的课程表（已删除的 discovery→transition
    桥）—— 实测两次：平均动作 std 0.37 → 1.00，薪资降档的瞬间策略退化成
    噪声。transition cfg 已删除；直接训练生产任务。

已烘焙进共享状态机的 run 教训（细节见 mdp.py 段头；每一条都花掉过一次
run 才学到）：
  • run-1：常开的 upright 奖励是一份站立薪资，压过一切冒险尝试（站着收
    了 4000 iter）；站立时足底接触传感器会 DROP OUT（100% 的回合实测），
    裸接触锁存会给站立付分 → apex 增量门（JUMP_MIN_APEX_GAIN）。
  • run-2：半目标 + 不可达空中参考的 max(air, dx) 付率让 2.9 cm 的微跳
    收走全部年金 → 付率只按动作质量对满目标计；踝门（JUMP_MIN_FOOT_GAIN）
    封掉"种植足摇摆"的假飞行。
  • run-3：单向落地锁把计分前沿冻在第一次微落地；锁存门槛必须落在防抖的
    弹道地板上（4 mm 躯干 = g·(0.06 s)²/8），不能压在策略典型采样跳的
    正上方。
  • run-4（用户评审"跳一次就站着不动，距离也短"）：每回合一次的距离付款
    + 永不关闭的年金让"跳一次然后冻结"成为连续任务的最优 —— 逐次付款 +
    会关闭的窗口修好了它，这就是 FrogJump。同一套经济学反过来正是两个单
    跳任务的定义，所以模式开关放在共享状态机里。
  • run-5：双足 RUN 物理上就是一串小弹道跳 —— 能通过所有飞行门；只有
    ≥0.35 s 的站姿节奏能把蛙跳和碎步跑分开（JUMP_MIN_STANCE_S）。

任务常数（JUMP_TARGET_X / JUMP_TARGET_Z）来自探针实测 ——
scripts/probe_jump.py 测了开环蹲伸动作（双脚离地 0.10-0.12 s，弧长约
2 cm）；RL 的闭环跳远远超它（2026-10 各 run 实测前向弧长达 8.5-11.9 cm
→ 目标定为 8 cm）。JUMP_TARGET_Z 受同一批飞行的起跳速度约束（v_z =
g·t/2 ≈ 0.49-0.59 m/s ⇒ apex 12-18 mm；RL 实测最大 rise 10.6 mm）——
最初 5 cm 的猜测不可达约 3 倍，暖启动策略卡在实测 6 mm 的天花板上；
2 cm 是"可达但有拉伸"的目标（见常数注释）。抬高前先做垂直方向探针。
"""

import math
import os
from copy import deepcopy

# 对称性 —— 三个动作都是矢状面/左右对称的；镜像损失专治侧向塌方
# （roulade 的先例）。
ENABLE_SYMMETRY = True

# ── 暖启动（配方见模块 docstring）─────────────────────────────────────────────
# 只要 MICRODUCK_WARM_START=1 就生效（mdp.py 里 Patch 5 读的同一个开关）——
# 一个开关管全部：计数器重置 + 所有课程从 step 0 钉在末段。全新 run 保持
# 正常爬坡。（velstand 的先例：它的生产 cfg 永久 WARM_START=True —— 这里
# 保留为运行开关，因为 FrogJump 本身是从头训出来的。）
WARM_START = os.environ.get("MICRODUCK_WARM_START", "") not in ("", "0")

# ── 域随机化（对齐 roulade/standup，保证 sim2real 一致）──────────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # 对齐 velocity（关）
ENABLE_KD_RANDOMIZATION              = False  # 对齐 velocity（关）
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # 跳跃途中来一推没有物理意义
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── 随机化范围（对齐 roulade 环境）───────────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # 课程爬坡到 0.015
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # 课程爬坡到 0.01
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # 未用（kp DR 关）
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # 未用（kd DR 关）
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# 回合长度。FrogJump 在 6 s 里装 3-4 跳（run-4 每次落地都付分，回合越长
# 装的跳越多）。单跳任务只付一次：下蹲 ~1 s + 飞行 + 落稳 —— 4 s
# （LongJump，run-1-3 的长度）和 5 s（JumpHigh；最大高度尝试需要多一拍
# 蓄力空间）。
EPISODE_LENGTH_S_FROG = 6.0
EPISODE_LENGTH_S_LONG = 4.0
EPISODE_LENGTH_S_HIGH = 5.0

# 实测站立躯干高度（standup 的教训：不许拍脑袋）。
STAND_Z = 0.115

# ── 任务常数 ──────────────────────────────────────────────────────────────────
# 跳远目标弧长（frog + long jump 共用）。测量历史和"后抬不先抬"规则见
# 模块 docstring。
JUMP_TARGET_X = 0.08
# 垂直跳目标：触地帧躯干 z 峰值 − 起跳 z。上限受实测起跳速度约束，不是
# 最初拍出来的弧长弹道换算（那个给了 5 cm）：探针的 0.10-0.12 s 双脚
# 飞行意味着 v_z = g·t/2 ≈ 0.49-0.59 m/s（apex 12-18 mm），2026-10 全部
# 跳跃的 RL 实测最大 rise 是 10.6 mm（v_z ≈ 0.46）—— 5 cm 的 rise 需要
# v_z ≈ 1.0 m/s，是这台机器人任何实测数据的两倍（5mm 错 STAND_Z 的教训：
# 不可达的目标会把任务变成不可能任务）。2 cm 和 JUMP_TARGET_X 一样是
# "可达但有拉伸"的赌注：探针的全垂直转向 + 更深的下蹲反击。等策略把
# 付款封顶了再按 run-4 的方式抬高，不要一上来就抬。
JUMP_TARGET_Z = 0.02
# 引导奖励的逐次飞行空中时间上限（0.35 s 双脚离地 ≈ 15 cm 弹道上升 ——
# 远超这些舵机能做出来的任何东西）。
JUMP_AIR_TIME_CAP_S = 0.35
# 起步势能（height_progress）的天花板：站立高度 + 一个好跳能超出站立
# 的高度余量。JumpHigh 抬高它，让势能在目标高度的飞行内仍然付分
# （0.115 + 0.02 = 0.135 apex；0.16 留了余量又不至于放开暴力区间）。
JUMP_APEX_CEILING = 0.15
JUMP_HIGH_APEX_CEILING = 0.16

# ── 舵机保护栈（权重/阈值照抄 velstand）──────────────────────────────────────
# 从 step 0 保持 25% 让梯度活着；PROTECT_FULL_STEP 时上满权重。
PROTECT_STAGE0_FRAC = 0.25
PROTECT_FULL_STEP   = 2000 * 24
HEAD_IMPACT_WEIGHT    = -0.01      # 头部子树每超 15 N 计一分
TRUNK_IMPACT_WEIGHT   = -0.01      # 躯干壳每超 20 N 计一分
SERVO_ACC_SPIKE_WEIGHT = -1e-3     # 每 rad/s² 超 300 计一分（舵机求和）
SERVO_STALL_WEIGHT    = -0.05      # 每个堵转舵机每步
GENTLE_LANDING_WEIGHT = 0.005      # 正数：trunk_vertical_accel_penalty 自带负号
HEAD_IMPACT_THRESH_N  = 15.0
TRUNK_IMPACT_THRESH_N = 20.0
SERVO_ACC_THRESH      = 300.0
SERVO_STALL_TORQUE    = 0.4
SERVO_STALL_VEL       = 0.5

# 落地复合姿态项用的腿舵机关节下标。仅限普通 standup（groundcontact）模型
# —— 这里舵机关节就是 0-13、没有被动关节交错，所以直接用 AGENTS.md 的
# 布局常数。将来如果注册 -Backlash- 跳跃变体，必须换成 _servo_joint_ids
# 辅助函数 / 按真实模型解析（backlash 模型上被动关节是交错的）。
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def _make_jump_base(
    play: bool = False,
    episode_length_s: float = EPISODE_LENGTH_S_FROG,
    multi_flight: bool = True,
    dx_ref: float = JUMP_TARGET_X,
    gain_ref: float = 0.0,
    apex_ceiling: float = JUMP_APEX_CEILING,
    collapse_smoothness: bool = True,
) -> ManagerBasedRlEnvCfg:
    """三个跳跃任务共享的全部内容：机器人、传感器、引导奖励、落地年金、
    坠落经济学、观测（61D 家族契约）、DR、终止、事件、地形和保护/平滑度
    课程。各任务的判别项（距离 vs 高度）和漂移税由公开工厂函数追加。

    ``dx_ref`` / ``gain_ref`` 选择落地年金的付率基准（跳远任务按前向
    弧长，JumpHigh 按 apex 增量 —— 见 mdp._jump_annuity_rate）；
    ``multi_flight`` 选择触地/窗口经济学（见 mdp.py 跳跃段头）。
    """

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # 头部/躯干对地力传感器 —— 坠落式保护（摔烂的跳会计费；脚先着地的跳
    # 永远碰不到阈值）。
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern=rf"^({'|'.join(HEAD_BODY_NAMES)})$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="^trunk_base$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── 基础配置 ──────────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, head_ground_cfg, trunk_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = episode_length_s

    # ── 动作 ──────────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── 奖励：删掉步行专用项 ──────────────────────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── 奖励：共享跳跃任务集 ──────────────────────────────────────────────────
    # 稠密发现引导：封顶的双脚离地时间。
    cfg.rewards["jump_air_time"] = RewardTermCfg(
        func=microduck_mdp.jump_air_time,
        weight=0.5,
        params={"air_time_cap_s": JUMP_AIR_TIME_CAP_S},
    )

    # 起跳引导：基于势能的高度进度，上限抬到站立之上 —— 下蹲→起跳的上升
    # 付一次分，上下晃动净零，摔跤后爬起来也照付（设计上不可刷分）。
    cfg.rewards["jump_launch"] = RewardTermCfg(
        func=microduck_mdp.height_progress,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ceiling": apex_ceiling,
        },
    )

    # 落地年金 —— 宽层 + 锐层（roulade run-4 教训：宽层够到，锐层收尾），
    # 再加站立税砸碎"落地后瘫成一团"的盆地。全部以落地锁存为门，并按
    # 付费飞行的动作质量对满目标缩放（run-1 教训：平价年金是站立就能收
    # 的 jack；run-2 教训：半目标的付率让 2.9 cm 微跳收走全部 —— 基准必须
    # 是满目标）。
    cfg.rewards["jump_landing"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_composite,
        weight=2.0,
        params={
            "target_height":  STAND_Z,
            "height_std":     0.04,
            "upright_std":    0.40,
            "pose_std":       0.45,
            "joint_indices":  _LEG_JOINTS,
            "dx_ref":         dx_ref,
            "gain_ref":       gain_ref,
        },
    )
    cfg.rewards["jump_landing_sharp"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_sharp,
        weight=1.0,
        params={
            "target_height": STAND_Z,
            "height_std":    0.015,
            "upright_std":   0.3,
            "dx_ref":        dx_ref,
            "gain_ref":      gain_ref,
        },
    )
    # 正数物理量 → 负权重（mjlab 成本约定）。
    cfg.rewards["jump_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.jump_stand_tax,
        weight=-4.0,
        params={
            "target_height": STAND_Z,
            "dx_ref":        dx_ref,
            "gain_ref":      gain_ref,
        },
    )

    # 坠落经济学（见 mdp.py 跳跃段）。正数物理量 → 负权重。
    cfg.rewards["jump_crash_tax"] = RewardTermCfg(
        func=microduck_mdp.jump_crash_tax,
        weight=-0.3,
        params={"gate_tilt_above_deg": 40.0},
    )

    # ── sim2real 正则项 ────────────────────────────────────────────────────────
    # action_rate 从 step 0 保留一个轻地板（run-4：连续蛙跳消耗的动作量是
    # 单跳的 ~5 倍 —— 该税绝不能和任务奖励分庭抗礼），再按课程爬坡。
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # 跳跃会俯仰，但要温和
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # 常开 upright 已删除（roulade 先例；run-1 教训）：权重 0.5 时光是站着
    # 就每步白拿 ~0.5 —— 一份旱涝保收的薪资，压过了冒险的年金，教会
    # run-1 站了 4000 iter。落地直立性由带锁存门的项计价。
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # 舵机保护栈 —— step 0 打 25%，PROTECT_FULL_STEP 上满。
    # ⚠️ gentle_landing 自带负号（返回 -|a_z|）→ 必须配正权重（罚项符号
    # 约定；配负权重等于奖励暴力）。
    _frac = PROTECT_STAGE0_FRAC
    cfg.rewards["head_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=HEAD_IMPACT_WEIGHT * _frac,
        params={"sensor_name": head_ground_cfg.name, "threshold": HEAD_IMPACT_THRESH_N},
    )
    cfg.rewards["trunk_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=TRUNK_IMPACT_WEIGHT * _frac,
        params={"sensor_name": trunk_ground_cfg.name, "threshold": TRUNK_IMPACT_THRESH_N},
    )
    cfg.rewards["servo_acc_spike"] = RewardTermCfg(
        func=microduck_mdp.servo_acc_spike_penalty,
        weight=SERVO_ACC_SPIKE_WEIGHT * _frac,
        params={"acc_thresh": SERVO_ACC_THRESH},
    )
    cfg.rewards["servo_stall"] = RewardTermCfg(
        func=microduck_mdp.servo_stall_penalty,
        weight=SERVO_STALL_WEIGHT * _frac,
        params={"torque_thresh": SERVO_STALL_TORQUE, "vel_thresh": SERVO_STALL_VEL},
    )
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=GENTLE_LANDING_WEIGHT * _frac,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # 自碰撞 —— 轻：深蹲伸的下蹲会把腿折向躯干（roulade 收腿先例）。
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── 观测（与步行/standup 策略完全相同的布局）─────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # 命令观测槽：head (4) 和 body (6) 都补零 —— 保住与 velocity/standup 的
    # 61D 观测布局一致性，运行时栈原样可用（发全零即可）。
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # Critic-only 特权高度观测（mdp.jump_height_state）：跳跃奖励全是高度
    # 重的（落地高度、apex 增量、站立税缺口）—— 价值函数直接拿到这些量，
    # 正如 velocity 的 critic 通过 ray 传感器拿到 foot_height。绝不进
    # actor（真机没有高度传感器；61D actor 契约是部署热插拔的基础）。
    # ⚠️ 2026-10-09 之前训练的 checkpoint critic 观测更短 —— mdp.py 的
    # Patch 6 会在形状不匹配时回退为仅加载 actor，而不是崩溃。
    cfg.observations["critic"].terms["jump_height_state"] = ObservationTermCfg(
        func=microduck_mdp.jump_height_state, scale=1.0,
    )

    # ── 命令：零附近微小噪声（保住观测形状一致性）───────────────────────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (episode_length_s, episode_length_s * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── 终止 ──────────────────────────────────────────────────────────────────
    # 跳失败是任务的一部分 —— 保留 NaN 卫兵、只看倾角的 1 s 摔跤回收
    # （限制废数据量；只看倾角放过深蹲）和超时。
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )
    cfg.terminations["fallen_recycle"] = TerminationTermCfg(
        func=microduck_mdp.fallen_too_long,
        params={
            "gate_z_below": 0.0,   # 只看倾角：深蹲是合法动作
            "gate_tilt_above_deg": 40.0,
            "max_duration_s": 1.0,
        },
    )

    # ── 事件 ──────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # 站立起点（随机偏航）+ 跳跃状态清零。必须注册在 reset_robot_joints
    # 之后（dict 插入序 = 执行序：关节先归位 HOME）。
    # multi_flight 把任务经济学写进状态机。
    cfg.events["set_jump_state"] = EventTermCfg(
        func=microduck_mdp.reset_jump_state,
        mode="reset",
        params={
            "z_min": 0.11,
            "z_max": 0.12,
            "tilt_max": math.radians(5.0),
            "joint_noise_std": 0.05,
            "multi_flight": multi_flight,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── 地形 ──────────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── 课程 ──────────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # 平滑度爬坡 —— 平滑度跟着技能走（standup/roulade 的时机教训）。run-4：
    # 连续蛙跳消耗的动作量是单跳的 ~5 倍，所以上限再砍半（−0.1）且来得晚
    # （2500）。
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.05},
                {"step": 2500 * 24,  "weight": -0.1},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,           "weight": 0.0},
                {"step": 2500 * 24,   "weight": -5e-4},
            ],
        },
    )

    # 保护爬坡：stage-0 比例 → 满权重（velstand 模式）。
    # gentle_landing（自带负号的函数）用正数段，mjlab 风格成本用负数段。
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,                "weight": GENTLE_LANDING_WEIGHT * PROTECT_STAGE0_FRAC},
                {"step": PROTECT_FULL_STEP, "weight": GENTLE_LANDING_WEIGHT},
            ],
        },
    )
    for _name, _full in [
        ("head_impact", HEAD_IMPACT_WEIGHT),
        ("trunk_impact", TRUNK_IMPACT_WEIGHT),
        ("servo_acc_spike", SERVO_ACC_SPIKE_WEIGHT),
        ("servo_stall", SERVO_STALL_WEIGHT),
    ]:
        cfg.curriculum[f"{_name}_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name":   _name,
                "weight_stages": [
                    {"step": 0,                 "weight": _full * PROTECT_STAGE0_FRAC},
                    {"step": PROTECT_FULL_STEP, "weight": _full},
                ],
            },
        )

    # 暖启动（配方见模块 docstring）：所有爬坡从 step 0 钉在末段 —— 加载
    # 的策略已经"穿过"这些爬坡训练过了。
    # ``collapse_smoothness=False`` 保留 action-rate/torque-rate 爬坡：一个
    # 需要在暖启动之上"重新发现"行为变体的任务（JumpHigh 把蛙跳的前向
    # 起跳改向垂直），如果在探索新变体的同时就背上巩固期 action_rate
    # （−0.1），会被自己的尝试税压死 —— 2026-10-09 实测（暖启动 run：最
    # 好的跳跃回合净收益 −0.21，光 action_rate 就 −0.28 vs 跳跃收入
    # +0.23；策略收敛到站立）。DR 和舵机保护栈照常折叠：加载的策略已经
    # 掌握它们且计费很小。
    if WARM_START and not play:
        from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
            _collapse_curricula_to_final,
        )

        if collapse_smoothness:
            _collapse_curricula_to_final(cfg)
        else:
            _smoothness = (
                "action_rate_weight",
                "torque_rate_weight",
            )
            _saved = {k: cfg.curriculum.pop(k) for k in _smoothness}
            _collapse_curricula_to_final(cfg)
            cfg.curriculum.update(_saved)

    return cfg


def make_microduck_frog_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Mjlab-FrogJump-Flat-MicroDuck —— 连续前向蛙跳。

    run-4 生产跳远配方，原样保留：每次合格触地支付自己那段封顶前向弧长
    （multi 模式），落地窗口逐跳打开、站立 JUMP_LAND_WINDOW_S 后关闭，付
    率跟随最后一次飞行的弧长 —— 连续跳好每一步严格最优（"应该一直跳"，
    创造这套经济学的 run-4 用户评审）。实验名保留 `microduck_jump`：已训
    练的 run-4 checkpoint（2500 iter，2026-10-09）就是这个任务的策略。
    """
    cfg = _make_jump_base(
        play=play,
        episode_length_s=EPISODE_LENGTH_S_FROG,
        multi_flight=True,
        dx_ref=JUMP_TARGET_X,
        gain_ref=0.0,
        apex_ceiling=JUMP_APEX_CEILING,
    )

    cfg.rewards["jump_distance"] = RewardTermCfg(
        func=microduck_mdp.jump_distance_bonus,
        weight=8.0,
        params={"target_x": JUMP_TARGET_X},
    )

    # 直线性 —— 向前跳，不斜不绕。连续蛙跳的 run-4 校准：连跳天然走弧线
    # 且每一跳都是剧烈动作，所以漂移税减半、容差放宽。
    cfg.rewards["jump_yaw_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_drift_penalty,
        weight=-0.2,
        params={"yaw_tol_rad": 0.8},
    )
    cfg.rewards["jump_lateral_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_lateral_drift_penalty,
        weight=-0.2,
        params={"lat_tol_m": 0.05},
    )

    return cfg


def make_microduck_long_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Mjlab-LongJump-Flat-MicroDuck —— 单次立定跳远，落地站稳。

    单跳模式（``multi_flight=False``）：每回合只有第一次合格触地支付封顶
    前向弧长，年金付率锁定在那次跳上，落地窗口永不关闭 —— "跳一次好的、
    落地、站到超时"按设计就是最优（run-4 之前的经济学，只对连续任务是
    错的）。距离权重保持 8.0：run-1-3 时代用这套付款形状训到了 7.6 cm 的
    中位弧长。

    漂移容差回到单跳标定（yaw 0.5 rad —— run-2 实测的 13.5° 中位 /
    20.8° p90 飞行偏航 —— 侧移 3 cm）：run-4 的放宽是给"连跳"的让步，
    本任务不连跳。不加前向漂移税：向前正是它拿钱的方向。
    """
    cfg = _make_jump_base(
        play=play,
        episode_length_s=EPISODE_LENGTH_S_LONG,
        multi_flight=False,
        dx_ref=JUMP_TARGET_X,
        gain_ref=0.0,
        apex_ceiling=JUMP_APEX_CEILING,
    )

    cfg.rewards["jump_distance"] = RewardTermCfg(
        func=microduck_mdp.jump_distance_bonus,
        weight=8.0,
        params={"target_x": JUMP_TARGET_X},
    )
    cfg.rewards["jump_yaw_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_drift_penalty,
        weight=-0.2,
        params={"yaw_tol_rad": 0.5},
    )
    cfg.rewards["jump_lateral_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_lateral_drift_penalty,
        weight=-0.2,
        params={"lat_tol_m": 0.03},
    )

    return cfg


def make_microduck_jump_high_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Mjlab-JumpHigh-Flat-MicroDuck —— 单次垂直跳，越高越好。

    单跳模式。判别项是 jump_height_bonus：合格飞行支付自己的 apex 增量
    （躯干 z 峰值 − 起跳 z）封顶 JUMP_TARGET_Z × 垂直度（以起落点水平位移
    为自变量的硬高斯门，mdp.JUMP_VERT_TOL_FINAL_M 为课程终值）—— 落地年金
    的付率基准换成同一个带门增量（``gain_ref``）。

    任务定义（用户，2026-10-10）：垂直向上跳 —— 原地起跳、原地落下、
    高度即得分。第一版只付裸 rise，策略交出了 6 cm 的斜跳（实测 85.7%
    侧移主导）：下面的漂移罚项想矫正它但弱了几个数量级 —— 正是
    AGENTS.md 那条教训的原文，动作定义要写进付款的门里，而不是小罚项。
    因此本任务不再注册三个漂移税项：门生效后它们是死重（2 cm 容差已经
    禁掉了斜跳，3 cm 的罚项根本看不见它）。

    起跳势能的天花板抬高（JUMP_HIGH_APEX_CEILING），让下蹲→起跳势能在
    目标高度的飞行内仍然付分。抬高目标走 run-4 路径（策略把付款封顶
    ⇒ 再抬），不提前抬。

    ⚠️ 经济学（2026-10-09 暖启动实验，见模块 docstring）：垂直跳是在蛙跳
    技能之上的行为"改向"，不是巩固 —— 尝试税必须拖着它。本任务在暖启动
    时保持平滑度课程"不折叠"，用自己的阶段（action_rate −0.02 地板 →
    −0.1 晚到），并以熵 0.01 探索：按蛙跳的巩固值（step 0 就 −0.1、熵
    0.003），最好的跳跃回合净收益 −0.21（光 action_rate 就 −0.28 vs 跳跃
    收入 +0.23），run 收敛到站立不动。
    """
    cfg = _make_jump_base(
        play=play,
        episode_length_s=EPISODE_LENGTH_S_HIGH,
        multi_flight=False,
        dx_ref=0.0,
        gain_ref=JUMP_TARGET_Z,
        apex_ceiling=JUMP_HIGH_APEX_CEILING,
        collapse_smoothness=False,
    )

    # 判别项 8.0 时被 dt 缩放稀释成噪声（满目标付款 = 8.0 × 0.02 = 0.16
    # 回合和 vs ~7.0 的年金）：20.0 时满目标跳在年金梯度之上多出看得见的
    # 0.4。×_jump_last_vert 的垂直度门在函数内部。
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=microduck_mdp.jump_height_bonus,
        weight=20.0,
        params={"target_z": JUMP_TARGET_Z},
    )

    # 尝试税跟着改向走（见上方经济学注释）：垂直蹲伸起跳是一个"新的"
    # 动作模式 —— 蛙跳巩固期的 step 0 就 −0.1，把起跳摆动收得比跳挣得还
    # 多。爬坡：探索地板 → 轻 → 家族上限。
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,           "weight": -0.02},
                {"step": 1000 * 24,   "weight": -0.05},
                {"step": 2000 * 24,   "weight": -0.1},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,           "weight": 0.0},
                {"step": 2000 * 24,   "weight": -5e-4},
            ],
        },
    )
    # 课程会在 step 0 改写实际权重；把 cfg 权重改成一致，stage-0 断言和
    # 课程生效前的步骤都不会打架。
    cfg.rewards["action_rate_l2"].weight = -0.02

    # 垂直度门的容差课程（写模块状态 mdp.jump_vertical_tolerance，状态机的
    # 垂直度高斯门读它）：先宽后紧。第 0 档压在暖启动策略的现有行为上
    # —— 蛙跳的习惯跳移动 4-9 cm，0.10 m 下"接近原地"的跳还能拿到部分
    # 分，改向的梯度保持活着；然后逐步收紧到生产 2 cm 门。step 0 就固定
    # 2 cm 实测：每个暖启动跳都 ≈0 分 → 又站回去了（2026-10-10，it571）。
    # ⚠️ 特意放在这里、_make_jump_base 之后添加：暖启动的课程折叠在 base
    # 内部运行，否则会把这个 (step, tol) 阶段表钉在末档 2 cm 上 —— 在
    # 恰恰需要宽门的那些 run 上复活零梯度悬崖。
    cfg.curriculum["jump_vertical_tolerance"] = CurriculumTermCfg(
        func=microduck_mdp.jump_vertical_tolerance_curriculum,
        params={
            "tol_stages": [
                {"step": 0,          "tol": 0.10},
                {"step": 500 * 24,   "tol": 0.05},
                {"step": 1000 * 24,  "tol": 0.03},
                {"step": 1500 * 24,  "tol": microduck_mdp.JUMP_VERT_TOL_FINAL_M},
            ],
        },
    )

    return cfg


# ── RL runner 配置 ────────────────────────────────────────────────────────────

def _jump_runner_cfg(
    experiment_name: str, run_name: str, max_iterations: int, entropy_coef: float = 0.003,
):
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,  # normalizer 必须由 export.py 烘进 ONNX
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=PpoWithSymmetryCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=entropy_coef,  # FrogJump 0.003 = run-4 巩固期值
            # （技能已在，探索已毕）。JumpHigh 覆写为 0.01：它的垂直跳是
            # 暖启动之上的"改向" —— 0.003 时采样 std 在 160 iter 内从
            # 0.23 塌到 0.08，探索在找到垂直变体之前就死了（2026-10-09
            # 暖启动实验）。
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
        ),
        wandb_project="mjlab_microduck",
        experiment_name=experiment_name,
        run_name=run_name,
        save_interval=250,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )


# FrogJump 保留历史实验名：run-4 生产 run（microduck_jump，截至 2026-10-09
# 已训 2500 iter）就是本任务的 checkpoint。
MicroduckFrogJumpRlCfg = _jump_runner_cfg("microduck_jump", "microduck_jump", 4_000)
# 单跳任务：每回合一个动作，从蛙跳技能暖启动 —— 预算取蛙跳的一半
# （AGENTS.md 预算：简单 episodic 动作 ≈ 4096 envs × 1000 iter；这里起点
# 更靠前）。JumpHigh 用更高的熵探索（见工厂里的经济学注释）：改向必须先
# 找到垂直变体，然后才谈巩固。
MicroduckLongJumpRlCfg = _jump_runner_cfg("microduck_long_jump", "microduck_long_jump", 2_000)
MicroduckJumpHighRlCfg = _jump_runner_cfg(
    "microduck_jump_high", "microduck_jump_high", 2_000, entropy_coef=0.01,
)
