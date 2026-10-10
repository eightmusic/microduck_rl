# -*- coding: utf-8 -*-
# ============================================================================
# 【学习副本 —— 不是训练用的真实配置】
#
# 本文件是 src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py 的中文详细
# 注释版，用于学习"Microduck 速度行走任务"的配置结构。
#
# ⚠️ 重要：
#   1. 训练仍然使用 src/ 下的原始文件，不要从 docs/ 导入本文件！
#      任务注册发生在 src/mjlab_microduck/tasks/__init__.py，它导入的是原始
#      模块；docs/ 目录不在 Python 包路径中，本文件永远不会被自动加载。
#   2. 本文件与原始文件可能逐渐不同步。学习时若发现数值与原始文件不一致，
#      以 src/ 下的原始文件为准。
#   3. 阅读顺序建议：先看每个【第 N 部分】的标题级注释建立全局结构，再深入
#      逐行注释。
#
# ============================================================================
# 【全文件总览 —— 先看这个再往下读】
#
# 这个文件回答 8 个问题（对应下面的 8 大类配置）：
#
#   问题 1：机器人长什么样，站在什么地方？      → 【第 5 部分】Scene 场景
#   问题 2：policy 输出什么控制量？             → 【第 6 部分】Action 动作
#   问题 3：policy 每一步看到什么？             → 【第 8 部分】Observation 观测
#   问题 4：机器人被要求完成什么？              → 【第 9 部分】Command 命令
#   问题 5：做得好/坏怎么打分？                 → 【第 7 部分】Reward 奖励
#   问题 6：一次尝试什么时候结束？              → 【第 11 部分】Termination 终止
#   问题 7：训练中环境如何变化（随机化/推扰）？ → 【第 10 部分】Events 事件
#   问题 8：难度如何随训练进度增加？            → 【第 12 部分】Curriculum 课程
#
# 除此之外还有：
#   【第 1 部分】模块级常量与开关：整个文件共用的"旋钮"
#   【第 2 部分】import：配置类型和函数从哪里来
#   【第 3 部分】rough terrain 地形生成配置
#   【第 4 部分】地形接触软化辅助函数
#   【第 13 部分】MicroduckRlCfg：PPO 训练配置（网络/算法/rollout）
#
# 【运行时三层关系 —— 理解本文件的关键】
#
#   ① 配置层（本文件）：只是"写菜谱"。执行 make_..._env_cfg() 时，
#      一行行往 cfg 对象里填 RewardTermCfg / ObservationTermCfg 等，
#      这些cfg对象只是记录"以后调用哪个函数、传什么参数、权重多少"。
#      此时还没有机器人、没有仿真、没有训练。
#
#   ② 环境构建层：训练开始时，mjlab 用这份 cfg 创建 ManagerBasedRlEnv，
#      各 manager（RewardManager、ObservationManager……）读取配置并
#      实例化每个 term。
#
#   ③ 运行层：训练循环中，每个 env.step(action) 会按固定顺序调用：
#      action_manager（处理动作）→ 物理仿真 → termination_manager
#      （判断结束）→ reward_manager（算分）→ reset → command/event
#      更新 → observation_manager（生成下一步输入）。
#      本文件配置的每个 term 就是在这个循环里被反复调用的。
#
# ============================================================================

# 【第 1 部分：模块级常量与开关】
# 这里的变量是"旋钮"：下面工厂函数组装配置时会读取它们。
# 改动它们的值 = 改变训练内容。大写命名是 Python 惯例，表示配置常量。

"""Microduck velocity (walking) environment.

The main locomotion task: velocity-command tracking + head-pose commands.
The reward/regularization recipe is locomotion-focused (lean tracking +
gait/feet terms, curriculum-ramped action-rate smoothing), with:

  - foot_slip kept at -0.1 (deliberately weak — stronger was too restrictive
    for this robot's pivot-heavy turning)
  - fixed, modest command ranges (ang ±1.0 makes turning learnable) instead of
    a widening curriculum that outpaced the robot's capability
  - turn-in-place: 15% of envs get lin=0 + |ang| ∈ [0.4, 1.0] (2026-07 audit:
    independent uniform sampling makes spin-on-the-spot ~2% of data → untrained)
  - head_pose_tracking as a primary objective, plus an EMA-based head_pose_bias
    penalty that prices only the escapable DC head droop (see below)
  - body_pose tracking infra kept intact but DISABLED (weight 0) so the obs
    slot stays alive for envs that use it
"""

import math          # 后面用于 math.sqrt（reward 的 std 参数）、math.log（质量随机化换算）
from copy import deepcopy  # 深拷贝。复制基础模板的配置对象后再修改，避免"共享引用"：
                     # 若多个任务/两个组（actor 和 critic）共享同一个可变对象，
                     # 改一处会悄悄污染另一处。这是本文件反复出现的防御手段。

# 每个 PPO iteration 中，每个并行环境采集多少个"控制步"。
# 注意：这不是 MuJoCo 物理步数！一个控制步内部还有 4 个物理子步（decimation=4）。
# 它同时也是 curriculum 中 step 的换算单位：curriculum 里的 "step" 是环境控制步，
# 例如 500 * 24 表示第 500 个 iteration。
NUM_STEPS_PER_ENV = 24

# 有 15% 的环境会被强制采样成"原地转身"命令（线速度=0，|yaw 角速度| ∈ [0.4, 1.0]）。
# 为什么需要：独立均匀采样几乎采不到"线速度≈0 且 |角速度| 很大"的组合（约 2%），
# 原地转身就等于没训练过 → 真机上转身又慢又不稳。这是"稀有但重要的命令需要
# 显式桶(bucket)"的教训。
TURN_IN_PLACE_FRACTION = 0.15

# 【左右镜像对称训练】开关。False = 不启用。
# 对非对称任务（如单手抓取）绝不能开；速度行走可用但当前未开。
ENABLE_SYMMETRY = False

# 【域随机化（Domain Randomization, DR）开关】
# DR = 每次环境重置/启动时随机扰动一些物理参数，让 policy 不依赖"完美固定的
# 仿真世界"，从而更容易迁移到真实机器人（sim2real）。
# True  = 工厂函数会把对应随机化事件登记进 cfg.events
# False = 不登记，训练中该参数保持仿真默认值
ENABLE_COM_RANDOMIZATION = True              # 躯干质心位置随机化
ENABLE_HEAD_COM_RANDOMIZATION = True         # 头部组件各 body 的质心随机化
ENABLE_KP_RANDOMIZATION = False              # 电机 P 增益随机化（曾经开过，现关闭）
ENABLE_KD_RANDOMIZATION = False              # 电机 D 增益随机化（同上）
ENABLE_MASS_INERTIA_RANDOMIZATION = True     # 质量+惯量一起缩放 ±5%
ENABLE_JOINT_FRICTION_RANDOMIZATION = True   # 关节摩擦随机化。注意 BAM 执行器下必须
                                             # 走 BAM 自己的 friction_scale 钩子（见第 10 部分）
ENABLE_JOINT_DAMPING_RANDOMIZATION = False   # 关节阻尼随机化。BAM 下 dof_damping 已被清零，
                                             # 开了也是无效操作（no-op），所以保持 False
ENABLE_ARMATURE_RANDOMIZATION = True         # 关节等效转子惯量随机化。BAM 会设置 armature
                                             # （不清零），所以这个随机化对 BAM 有效
ENABLE_VELOCITY_PUSHES = True                # 训练中随机"推一把"机器人，练抗扰
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # IMU 安装误差模拟：观测层面给 actor 看一个
                                             # 随机旋转了几度的 IMU 坐标系
ENABLE_ENCODER_BIAS = True                   # 编码器偏差：每个环境一个固定的小角度偏差，
                                             # actor 的关节位置观测里带上这个偏差（模拟真实
                                             # 编码器校准误差）；critic 用真实值
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False # 重置时随机倾斜机器人初始姿态（当前关闭）

# 头部/身体姿态命令的重新采样间隔（秒）。
# 命令不是每步都换：每隔 2~5 秒换一个新目标，policy 有时间执行到位。
# （替代了旧的"脖子偏置扰动"方案。）
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# 观测选择开关：True = 用 projected_gravity（重力方向投影到机身坐标系，3 维单位向量），
# False = 用 raw_accelerometer（原始加速度计读数）。
# 两者信息相近但数值含义不同，切换会改变 policy 的输入分布，训练/部署必须一致。
USE_PROJECTED_GRAVITY = True

# 下面是各类随机化的数值范围（工厂函数按开关取用）。
COM_RANDOMIZATION_RANGE = 0.003  # 躯干质心随机化初始 ±3mm，之后由 curriculum 逐步加大
# 头部质心随机化：对头部组件的每个 body 都加一个每次重置时新采样的偏移。
# 头-滚转 body 在 walk 模型里叫 bottom_head_shell，在 roller 模型里叫 jaw_soft，
# 所以名字列表里用正则交替匹配。
# ⚠️ 历史遗留：bearing_roll 其实不是头部部件（它是右髋 yaw 连杆），一直是列表里的
# 错误，为了保持已有 DR 行为不变才保留。
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # 头部质心初始 ±3mm，由 curriculum 逐步加大
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # 质量与惯量同时缩放 ±5%（两者绑定，
                                                 # 保证物理一致性；见第 10 部分 pseudo_inertia）
KP_RANDOMIZATION_RANGE = (0.85, 1.15)            # P 增益 ±15%（当前未启用）
KD_RANDOMIZATION_RANGE = (0.9, 1.1)              # D 增益 ±10%（当前未启用）
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)  # 关节摩擦预算缩放 ±10%
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)   # 关节阻尼缩放 ±10%（当前未启用）
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)        # 转子等效惯量缩放 ±10%
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)            # 推扰事件：每 3~6 秒推一次
VELOCITY_PUSH_RANGE = (-0.3, 0.3)                # 推扰给机器人的平面速度变化 ±0.3 m/s。
# 曾用 ±0.5：每 3~6 秒给一个比最大行走速度(0.4)还大的冲击，把 policy 训成了"永远
# 惊恐的跌倒恢复步态"。±0.3 保留抗扰能力，同时允许更从容的步态。
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # IMU 安装误差最大 6°。注意是"随机轴"零均值
# 的：训练的是对安装误差"大小"的容忍，不能补偿系统性的固定安装偏置——真机上有约 5°
# 的系统性 pitch 偏移，那是在 runtime 侧（imu-pitch-offset）修正的，不在这里。
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # 每关节编码器偏差 ±0.015 rad ≈ ±0.86°（每个环境固定一个值）
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # （未启用）重置时前后倾斜最大 10°
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0    # （未启用）重置时左右倾斜最大 5°

# 【第 2 部分：import —— 配置类型与函数的来源】
# 这里 import 的不是"实现"本身，而是"描述配置"的类。记住分工：
#   XxxCfg 类     = 配置对象（记录调用什么函数、参数、权重）
#   mdp.xxx 函数  = 运行时被 manager 实际调用的函数
# 两个同名 mdp 很容易混淆，务必区分：
#   mdp           = mjlab 上游通用速度任务自带的函数（基础模板）
#   microduck_mdp = 本项目自己写的函数（Microduck 特有逻辑）
import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg          # 本文件函数的返回类型：完整环境配置
from mjlab.envs.mdp import dr                        # mjlab 自带的域随机化函数库
from mjlab.envs.mdp.actions import JointPositionActionCfg  # "关节位置目标"动作的配置类
from mjlab.managers import (
    CurriculumTermCfg,   # 课程配置：随训练进度改变其他配置
    EventTermCfg,        # 事件配置：重置/随机化/推扰等
    ObservationTermCfg,  # 单个观测项配置：函数+参数+噪声+延迟
    RewardTermCfg,       # 单个奖励项配置：函数+参数+权重
    TerminationTermCfg,  # 单个终止条件配置
)
from mjlab.managers.scene_entity_config import SceneEntityCfg  # "选机器人里的哪些
                                                               # 关节/body/geom/site"的
                                                               # 选择器，广泛用于筛选
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,  # PPO 训练运行器的配置类型（文件末尾用）
    RslRlModelCfg,           # actor/critic 网络的配置类型
)
from mjlab.sensor import (
    ContactMatch,            # 接触传感器：匹配"哪两个东西之间"的接触
    ContactSensorCfg,        # 接触传感器配置
    ObjRef,                  # 引用模型里的某个对象（site/body…）
    RingPatternCfg,          # 射线采样图案（脚部高度传感器用）
    TerrainHeightSensorCfg,  # 地形高度（射线）传感器配置
)
from mjlab.tasks.velocity import mdp                          # mjlab 通用速度 MDP 函数
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # 速度命令配置类
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg  # ★上游基础模板工厂
from mjlab.utils.noise import UniformNoiseCfg as Unoise       # 均匀分布观测噪声配置

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG  # walk 机器人
from mjlab_microduck.tasks import mdp as microduck_mdp         # ★本项目自己的 MDP 函数集
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG  # 镜像对称训练


# 【第 3 部分：rough terrain 地形生成配置】
# 只有 rough=True 的环境会用它。Microduck 身高 ~25cm、抬脚只有 1~2cm，
# 所以地形比 mjlab 默认（台阶 10cm）温和得多。
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),      # 每块地形 8m × 8m
    border_width=20.0,    # 边界宽 20m（防止机器人走出有效区域）
    num_rows=10,          # 10 行：不同难度等级（curriculum 用）
    num_cols=20,          # 20 列：不同地形类型
    sub_terrains={        # 子地形种类及比例（proportion 总和为 1）
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),  # 25% 平地
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,                  # 25% 金字塔楼梯
            step_height_range=(0.0, 0.015),   # 台阶最高 1.5cm（默认是 10cm，对它太高）
            step_width=0.15,                  # 台阶深 15cm
            platform_width=2.0,               # 顶部平台 2m
            border_width=1.0,
        ),
        # 注：倒金字塔（下坡坑）地形被移除了——它把 env_origin_z 设到坑底（负值），
        # 重置位置 root_z = 0.12 + env_origin_z ≈ -0.10m，机器人出生在坑底之下
        # 直接穿地。这类"出生即失败"的地形必须排除。
        # 随机高低格（类似碎石路）：每个格子随机高度偏移。
        # grid_width=0.12 会让 8m 地块变成 66×66≈4356 个盒子 → 总共 26 万 → 显存爆。
        # 0.45m → 17×17=289 盒/块 → 约 1.7 万个，可以。
        # 另外格子宽必须不能整除地块尺寸（0.45×17=7.65 ✓），避免边缘切一半。
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,                  # 30% 随机格
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),   # 高低差最大 1cm
            platform_width=1.5,
        ),
        # 缓坡：heightfield 金字塔，平台在【顶部】——机器人出生在平台平地上，
        # 随命令重采样在坡上上下/横穿行走。
        # slope_range 是坡度(rise/run)：0.03→0.10 ≈ 1.7°→5.7°（小机器人小坡）。
        # 同样不取倒置版本（同上面的坑出生问题）。
        # vertical_scale=0.001 让高度量化步长为 1mm，缓坡是平滑的而不是 5mm 一级的台阶。
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,                  # 20% 缓坡
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,  # 不给生成的地形加灯光
)


# 【第 4 部分：地形接触软化辅助函数】
# rough 模式下由 cfg.scene.spec_fn 调用（见第 11 部分 terrain 小节）。
# 作用：把地形盒子的接触参数调软，抑制"脚落在两个不同高度盒子的硬棱上"
# 导致的接触法向不稳定 → 求解器冲出 NaN 力 → 训练崩溃。
def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """Soften terrain box geom contacts to reduce edge-contact NaN instability.

    Box terrains place adjacent geoms at different heights. The hard edges where
    heights change cause contact normal instability when feet land on them, which
    can produce impulsive NaN forces in the MuJoCo solver.

    Doubling the solref time constant (0.02 → 0.04 s) makes contact springs
    2× softer — enough to damp the instability without noticeably changing the
    macro-level walking physics. Applied to all geoms in the "terrain" body,
    which contains every box generated by TerrainGenerator.
    """
    # spec.body("terrain")：拿到地形生成器创建的名为 terrain 的 body（所有盒子都在它下面）
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:          # 逐个遍历地形盒子
        geom.solref = [0.04, 1.0]    # 接触弹簧时间常数 0.02→0.04（软 2 倍）
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # 阻抗参数也调软一点
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


# ============================================================================
# 【核心工厂函数】
#
# def make_microduck_velocity_env_cfg(play=False, rough=False) -> ManagerBasedRlEnvCfg
#
# 参数：
#   play  = False → 训练用配置（完整 curriculum、正常推扰间隔）
#           True  → play/viewer 用配置（推扰更频繁便于观察、地形网格更小等）
#           注意：play 不是"用不用 policy"，训练和 play 都用 policy；
#                 它表示"这份配置给训练过程用，还是给人看/验证用"。
#   rough = False → 平地
#           True  → 用【第 3 部分】的生成地形
#
# 返回值 ManagerBasedRlEnvCfg：只是一个 Python 配置对象。调用它不创建机器人、
# 不开仿真、不训练。真正的环境由训练脚本用这个配置去构造。
#
# 【内部执行顺序】（配置组装顺序，≠ 运行时顺序）：
#   ① 局部小配件（姿态 std、脚名、传感器配置）      → 行 A
#   ② 拿上游基础模板并换上 Microduck 机器人/传感器  → 行 B
#   ③ action 配置                                   → 行 C
#   ④ reward 配置                                   → 行 D
#   ⑤ events（重置/推扰/域随机化）+ termination     → 行 E
#   ⑥ observation 配置                              → 行 F
#   ⑦ command 配置                                  → 行 G
#   ⑧ head/body 姿态 reward                         → 行 H
#   ⑨ terrain（平地/rough 分支）                     → 行 I
#   ⑩ curriculum（难度课程）                        → 行 J
#   ⑪ return cfg
# ============================================================================
def make_microduck_velocity_env_cfg(
    play: bool = False,   # 是否生成 play（观看/验证）模式配置
    rough: bool = False,  # 是否使用崎岖地形
) -> ManagerBasedRlEnvCfg:
    """Create Microduck velocity tracking environment configuration."""

    # ------------------------------------------------------------------
    # 行 A：局部小配件（在配置 reward/传感器时使用）
    # ------------------------------------------------------------------

    # 站立姿态下各关节的"容忍度 std"——之后填进 pose reward。
    # 键是正则，匹配关节名；值是该关节组姿态 reward 的 std 参数（单位 rad）。
    # 站立时 std 小 = 希望关节贴近 HOME 姿态（command=0 时 reward 收得紧）。
    std_standing = {
        # 下肢
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — 保持 5° 内八站姿（脚掌平贴地面），防止腿外撇
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }

    # 行走时的姿态容忍度：腿部必须摆动，所以 std 更大（允许偏离 HOME 更多）。
    # ⚠️ 这个 std 是"pose reward 对误差的容忍度"，不是 policy 的探索噪声！
    std_walking = {
        # 下肢
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.05,  # 同样保持内八站姿，防止走路时腿甩成竖直
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,  # 原为 0.15
    }

    # 左右脚的 site 名（site = 模型里的参考点）。后面脚部传感器/脚部 reward 复用。
    site_names = ["left_foot", "right_foot"]

    # 脚-地接触传感器配置：
    # 提供"左脚/右脚是否接触地面、接触力多大、脚离地多久"的数据。
    # 消费者：foot_air_time/foot_slip 等 reward、critic 的 foot_contact* 观测、
    #        nan_state 终止检查。
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",  # 传感器名，其他配置按这个名字引用
        primary=ContactMatch(        # primary = 接触的"这一侧"是谁
            mode="geom",             # 按几何体(geom)匹配
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # 只匹配两只脚的碰撞几何
            entity="robot",          # 在名为 robot 的实体里找
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),  # 接触的"另一侧"= 地面
        fields=("found", "force"),  # 传感器输出：是否接触 + 接触力
        reduce="netforce",          # 力按合力汇总（不保留每个接触点）
        num_slots=1,                # 每只脚 1 个记录槽
        track_air_time=True,        # 额外记录"脚离地时长"（摆动相分析用）
    )

    # 自碰撞传感器：检测机器人自身部件之间是否发生接触。
    # primary/secondary 都指向 trunk_base 子树 = 检测躯干子树内部/与其他腿部的碰撞。
    # 消费者：self_collisions 惩罚 reward。
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),  # 躯干子树
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),  # 同上
        fields=("found",),  # 只需要"是否发生接触"，不需要力
        reduce="none",      # 不合并
        num_slots=1,
    )

    # 脚部离地高度传感器（射线式）：
    # mjlab 1.3.0 起，foot_height 观测和 foot_clearance/foot_swing_height 奖励都由
    # 这个"每只脚向下打射线测地形高度"的传感器驱动（旧版用 site 坐标）。
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",  # 传感器名
        # 对 site_names 里的每只脚建一个测量系（site 位置）
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),  # 半径 4cm 圈上采 2 个点
        ray_alignment="yaw",   # 射线按 yaw 方向对齐
        max_distance=1.0,      # 射线最长测 1m
        exclude_parent_body=True,    # 不把脚自己的父 body 当作命中对象
        include_geom_groups=(0,),    # 只测 geom group 0（默认碰撞组）
        debug_vis=False,             # viewer 里不画射线
    )

    # 脚部碰撞几何名列表：之后 foot_friction 随机化事件用。
    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # ------------------------------------------------------------------
    # 行 B：继承上游基础模板 + 换上 Microduck 的机器人和传感器
    # ------------------------------------------------------------------

    # ★ 关键一步：先拿 mjlab 的通用速度环境配置。
    # 它已经包含基础的 observation/action/command/reward/termination/curriculum。
    # 本文件接下来的一切都是"在这个基础上修改"。
    # 所以有些 term 在本文件里看不到定义（如 track_linear_velocity 的函数体），
    # 它们来自上游模板。
    cfg = make_velocity_env_cfg()

    # 换机器人：把模板里的实体换成 Microduck 的 walk 模型。
    # "robot" 是实体名，后面所有 SceneEntityCfg("robot") 都按这个名字找它。
    # MICRODUCK_WALK_ROBOT_CFG（定义在 microduck_constants.py）：
    #   - 加载 robot_walk.xml
    #   - 初始姿态 HOME_FRAME
    #   - 14 个主动关节全部挂 BAM 执行器（电压控制 XL330 舵机模型）
    #   - 关节软限位缩到 90%
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}

    # 装传感器：把行 A 定义的三个传感器配置放进场景。
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)

    # viewer 相机跟随躯干
    cfg.viewer.body_name = "trunk_base"

    # ------------------------------------------------------------------
    # 行 C：Action —— policy 输出的 14 维是什么
    # ------------------------------------------------------------------
    # 模板里的 action 项叫 "joint_pos"（关节位置目标型动作）：
    #   policy 输出 14 维 → 加上 scale 和默认姿态偏移(use_default_offset) →
    #   成为 14 个关节的位置目标 → BAM 执行器算实际力矩。
    # ⚠️ 14 = 主动舵机数。模型里可能有更多关节（passive_*），它们不占 action 维。
    # 训练新行为（跳跃/起身/翻滚）时通常不改这里——14 个舵机是所有任务共用的
    # 控制接口；新任务改的是 command/reward/termination/curriculum。
    joint_pos_action = cfg.actions["joint_pos"]
    # 断言类型：万一上游模板改成别的动作类型，这里立刻报错而不是悄悄改错对象
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    # scale=1.0：policy 输出 0.2 → 目标角 = HOME 姿态 + 0.2 rad
    #（模板默认 0.5，Microduck 用满幅 1.0 —— 低 kp 舵机需要更大行程和超调空间）
    joint_pos_action.scale = 1.0

    # ==================================================================
    # 行 D：Reward —— 每个 reward term 都在回答"这一步做得好不好"
    #
    # RewardTermCfg 的通用结构：
    #   func   = 运行时每个控制步被 RewardManager 调用的函数
    #   weight = 该函数返回值乘的权重（正=鼓励，负=惩罚）
    #   params = 传给 func 的参数（命令名、std、作用于哪些关节/身体…）
    # 运行时总 reward = Σ(每个 term 的函数值 × weight)，再乘环境步长 dt。
    #
    # ⚠️ 符号约定陷阱（本项目最著名的坑）：
    #   - mjlab 基础 cost 函数返回 ≥0        → weight 用【负】数
    #   - microduck 自带 penalty 返回 ≤0     → weight 用【正】数
    #   弄反 = 惩罚变奖励，policy 会去"刷"这个错误项。
    #   无可辩驳的检查：wandb 里每个 penalty 项的 Episode_Reward 必须 ≤ 0。
    # ==================================================================

    # --- pose 姿态奖励：腿部关节保持在合理姿态附近 ---
    # std_standing/std_walking 是按命令速度分档的容忍度（command=0 用站立档，
    # 行走用行走档——在 reward 函数里根据速度阈值切换）。
    cfg.rewards["pose"].params["std_standing"] = std_standing  # 命令=0 时用这组（收得紧）
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking    # Microduck 没有跑态，复用行走档
    # pose 只作用在【腿】上。头部/脖子由 head_pose_tracking 驱动——如果这里也管
    # 头部，pose 会把头拉向 HOME，而 head_pose_tracking 把头拉向命令，两者打架时
    # policy 会学会"无视头部命令"（因为大命令下 tracking 梯度衰减，pose 占上风）。
    # 正则解释：匹配所有"非 passive_、且名字不含 neck/head"的关节 = 纯腿关节。
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    # 行走判定阈值：root 速度超过 0.01 才算"在走"（切 std_walking 那组参数）
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # --- upright 直立奖励：躯干别歪 ---
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)  # 测躯干
    # 刻意加强（2.0 / std²=0.05，旧值 1.0 / std²=0.1）：
    # 2026-07 评测发现 policy 以 +2~4° 持续前倾走路（p90 达 6-8°），且速度下被推倒的
    # 2/3 是向前摔。旧参数下 4° 前倾每步只损失 ~0.05 分，等于免费；新参数下损失
    # ~0.19/步，足以让稳态步态保持躯干水平，同时瞬态前倾（抗推、加速）仍然划算。
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # --- 脚部几何选择：clearance/slip 用 site 选脚（swing_height 在 1.3.0 已改为纯传感器驱动） ---
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # body_ang_vel 惩罚的对象：躯干
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # foot_slip 刻意弱（-0.1 而不是 -1.0）：这只机器人转身靠大量枢轴旋转（pivot），
    # 脚在地上原地拧转是正常动作，罚太狠会把转身罚没了。
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01  # 命令速度高于此才计（站立不罚）

    # 删掉模板里的 soft_landing（着陆缓冲奖励）——速度行走任务不需要。
    cfg.rewards.pop("soft_landing", None)

    # --- 自碰撞惩罚：腿别撞躯干/电池盒 ---
    # 撞上 self_collision 类 geoms（腿、电池支架上的专用碰撞体）就罚。
    # 正常限位下其实碰不到，但保留一个正向"远离"信号更保险。
    # 注意 mjlab 的 self_collision_cost 返回 ≥0 → weight 必须是负数。
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,   # mjlab 上游函数
        weight=-1.0,                    # 负权重 = 惩罚
        params={"sensor_name": self_collision_cfg.name},  # 用上面定义的自碰撞传感器
    )

    # --- air_time 摆动相奖励：让每只脚的离地时间落在 [0.125, 0.300] 秒窗口 ---
    # 这个窗口塑造"迈步节奏"：太短的拖步和太长的跳跃式迈步都拿不到满分。
    # ⚠️ 零命令下的"站着别动"不靠这里教——那是 standing_envs curriculum 的活
    #（到 ~iter 2000 站立环境比例升到 25%），不要另加"禁止原地踏步"的项。
    cfg.rewards["air_time"].weight = 3.0            # 权重 3：相当强，步态塑形主力之一
    cfg.rewards["air_time"].params["command_threshold"] = 0.01  # 命令>0.01 才生效（站立不计）
    cfg.rewards["air_time"].params["threshold_min"] = 0.125  # 离地 <0.125s 拿不满分
    cfg.rewards["air_time"].params["threshold_max"] = 0.300  # 离地 >0.300s 也不加分

    # --- 抗"甩动"正则：躯干角速度惩罚 + 全身角动量惩罚 ---
    # 都属于"运动阻断器"（penalize what dynamic motion physically requires），
    # 行走任务保持低权重，别把步态需要的摆动罚没了。
    cfg.rewards["body_ang_vel"].weight = -0.05      # 躯干角速度大小 → 罚
    cfg.rewards["angular_momentum"].weight = -0.02  # 全身角动量 → 罚

    # --- 速度跟踪奖励：任务主目标 ---
    # track 函数（上游）返回 exp(-误差²/std²)：误差 0 → 1 分，误差越大 → 趋向 0。
    # std=sqrt(0.1)≈0.316 m/s：误差 0.316 m/s 时得分降到 1/e。这是"线性速度误差
    # 还在意"的尺度，不要设太松否则小误差没有梯度。
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    # 角速度 std 更大(≈0.707)：转身时身体必然摆动/滞后，容忍度比平移速度宽。

    # --- action_rate 动作变化率惩罚（平滑性正则） ---
    # 惩罚相邻两步 action 的差值平方和 → 防抖动。
    # 这里只写"初始值"：真正的权重由下面的 curriculum 从 -0.1 一路加到 -1.0
    #（iter 1500 时到位）。先轻后重的原因：技能发现期任何"尝试税"都会让
    # "什么都不做"成为最优解——步态还没学会就重罚动作变化 = 学不会走。
    cfg.rewards["action_rate_l2"].weight = -0.1

    # --- 抬脚高度目标：2cm ---
    # 之前 1cm 会导致拖脚（dragging），提高到 2cm 强制把脚抬起来。
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02  # 从 0.01 提高以惩罚拖脚

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02  # 同上，强制抬脚

    # 注意：没有单独的"脖子 action-rate"项——共享的 action_rate_l2 对全部 14 维
    # （含脖子）求和，而下面的 head_pose_tracking 已给 4 个脖子/头部自由度一个
    # 位置目标，脖子已被完全塑形，不需要额外项。

    # ==================================================================
    # 行 E：Events —— 什么时候发生什么"外部事件"
    #
    # EventTermCfg 的关键参数：
    #   mode="startup"  环境创建时执行一次
    #   mode="reset"    每个环境 episode 重置时执行（DR 大多在这）
    #   mode="interval" 以随机间隔（interval_range_s）反复执行（推扰在这）
    #   mode="step"     每个控制步执行
    # ==================================================================

    # BAM 摩擦字段扩展（startup）：
    # BAM 执行器每步写 per-env 的 dof_frictionloss/dof_damping，这个"无操作"事件
    # 的作用是把这几个字段登记为 per-world 展开（mjlab 的 per-env 字段机制需要注册）。
    # ⚠️ 所有独立环境配置必须注册它（AGENTS.md 不变量）。
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,  # 本项目自己的函数
        mode="startup",
    )

    # reset 时清空动作历史（reset）：episode 开始时把上一步 action 缓冲清零，
    # 避免上一条命的 action 泄漏进新 episode 的 observation。
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    # 脚底摩擦随机化：模板里已有 foot_friction 事件，这里只改参数——
    # 指定随机化哪两个几何体（两只脚）+ 收紧范围到 (0.7, 1.3)。
    # 曾是 (0.3, 1.2)：太滑的脚训练不出来，收紧成"更抓地"的范围。
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # --- NaN 终止：物理爆炸的保险丝 ---
    # MuJoCo 接触求解在极端冲击下可能产生 NaN 关节位置。NaN 一旦进入 observation
    # 会污染网络权重、毁掉整个训练。这个终止在检测到非有限状态时立刻重置该环境，
    # 让 NaN 来不及扩散。它检查关节/根部状态 + 脚部接触力（力可以先于状态炸）。
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,  # 本项目函数（见 mdp.py）
        time_out=False,      # False = 这是"真终止"（算失败），不是超时截断
        params={"sensor_names": (feet_ground_cfg.name,)},  # 顺带检查脚接触力传感器
    )

    # 重置时躯干初始高度：模板默认 (0.01, 0.05) 太低，Microduck 站立高约 0.12m，
    # 改成 (0.12, 0.13) = 从站立高度附近出生。
    # ⚠️ 此类高度必须实测模型（AGENTS.md 教训：5mm 的错让目标不可达）。
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # --- 推扰事件（interval 模式）：随机时刻给机器人一个平面速度冲击 ---
    if ENABLE_VELOCITY_PUSHES:
        # play 模式用更短的推扰间隔（0.5~1s）让观看者更容易看到抗扰行为
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,  # 上游函数：直接改写根速度
            mode="interval",                    # 随机间隔触发
            interval_range_s=interval,          # 训练 3~6s / play 0.5~1s
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,   # ±0.3 m/s
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # --- 域随机化（DR）说明 ---
    # 大部分 DR 在 mode="reset" 执行 = 每个 episode 重新采样，是"非累积"的：
    # mjlab 1.3.0 的 dr.* 在 operation="add"/"scale" 时每次重置都从编译期默认值
    # 重新计算（uses_defaults=True），不会越加越大。
    # （历史上本项目吃过累积式 CoM 随机化的亏，运行几个月才发现劣化。）

    # 躯干质心随机化（reset）：每 episode 给 trunk_base 质心加 ±3mm（初始）偏移。
    # 模拟真实机器人的质量分布误差。
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,        # 上游 DR 函数：body 的 ipos（局部质心坐标）
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",   # 在默认值上加随机偏移（非累积）
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    # 头部组件质心随机化（reset）：对 HEAD_BODY_NAMES 里的每个 body 分别加偏移。
    # 头占了全机 38% 质量，位置误差影响巨大，必须随机化。
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

    # 电机 PD 增益随机化（当前关闭）：使用能处理 DelayedActuator 的自定义函数。
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

    # 质量+惯量随机化（startup：整个训练期间每个环境固定一个值）：
    # 用上游 pseudo_inertia：以 alpha 同时缩放质量和惯量 e^(2·alpha)，质心不动
    # （所以不会和 randomize_com 打架）。alpha_range 由 ±5% 质量范围换算：
    # e^(2·alpha) ∈ [0.95, 1.05]。
    # ⚠️ 为什么不用直接写 body_mass/body_inertia 的旧函数：mjlab 1.3.0 下直接
    # 写不是 per-env 展开的，会静默塌缩成所有环境共享一个值 = 无效随机化。
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",   # 启动时固定 per env（质量 DR 的标准做法，无累积问题）
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    # 关节摩擦随机化（reset）：BAM 专用路径！
    # BAM 执行器把 MuJoCo 的 dof_frictionloss 清零了，直接随机化 dof_frictionloss
    # 是【静默无效操作】。必须缩放 BAM 自己的摩擦预算（Coulomb+Stribeck+负载项），
    # 走 FrictionDRBamActuator.friction_scale 钩子。
    # （AGENTS.md 不变量：BAM 下 joint-friction DR 必须缩放 actuator 的 friction_scale）
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,  # 本项目 BAM 专用函数
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,  # ±10%
            },
        )

    # 关节阻尼随机化（当前关闭）：即使开了，BAM 下 dof_damping 已被清零，
    # 只对旧的 XML position actuator 有效——留着是给非 BAM 路径用的。
    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # domain_randomization=True 时必填
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    # 关节等效转子惯量（armature）随机化（reset）：±10%，microban 同款。
    # 非"累积"（uses_defaults）。BAM 下有效——BAM 设置 dof_armature（≈0.0018），
    # 没有清零，所以缩放有效。
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

    # IMU 安装误差不是在这里（事件层）随机化的——旧的事件方案写 site_quat 在
    # mjlab 1.3.0 下既不 per-env 展开也不被观测读取 = 无操作。现在改在
    # 【观测层】实现（见行 F：把 projected_gravity / base_ang_vel 的函数换成
    # 带随机旋转的版本）。

    # 初始姿态倾斜随机化（reset，当前关闭）：重置时给机器人 ±10° pitch / ±5° roll，
    # 逼 policy 学会"从不正的姿态恢复"。
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # ==================================================================
    # 行 F：Observation —— policy 每一步"看到"什么
    #
    # ObservationGroupCfg = 一组观测（"actor"组和"critic"组），
    # ObservationTermCfg  = 组里的一个观测项（函数+参数+噪声+延迟）。
    # 运行时每个控制步，ObservationManager 对每项调用其函数，按声明顺序
    # 拼接成向量 → actor 输入 61 维，critic 输入 76 维。
    #
    # ★ 61D 观测布局是全项目策略族的部署契约（热切换前提），顺序不能变：
    #   [本体感觉 48] + [twist(3) + head_pose(4) + body_pose(6)] = 61
    #   本体感觉 48 = 角速度3 + 重力3 + 关节位置14 + 关节速度14 + 上步action14
    # 不使用的命令槽【零填充保留】，绝不删除槽位。
    # ==================================================================

    # --- 删减模板观测 ---
    # 删掉 actor 的 base_lin_vel（真实机体上没有可靠的线速度传感器）。
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 默认给两组都加 height_scan（机载地形射线扫描）。
    # Microduck 没有这种机载传感器，两组都删。
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # base_lin_vel 只加给 critic = "特权信息"：
    # critic 只在训练时估值用，可以看真实机器人拿不到的量；
    # actor 看不到 → 部署时也不需要。这是 asymmetric actor-critic 结构。
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # 重力观测名按开关选择（projected_gravity 或 raw_accelerometer，见第 1 部分）
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    if not USE_PROJECTED_GRAVITY:
        # 开关为 False：删掉重力项，换成原始加速度计读数
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    # 接下来要改这两个 term 的属性，先 deepcopy 隔离（模板里 actor/critic 可能共享
    # 同一个 term 对象，直接改会串味到 critic）。
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # IMU 数据延迟模拟：允许 0~1 个控制周期的延迟（0~20ms 封套），每 64 步换一次
    # 当前延迟。曾允许 3 步（60ms），2026-07 实测真实 dxl IMU 路径很快，收紧到 1。
    # （训练"带延迟"，policy 才不会依赖不存在的即时反馈。）
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # critic 的传感器类观测换成 "_safe" 包装：
    # nan_state 终止只检查关节+根部状态，而这三项读的是射线/接触传感器数据，
    # MuJoCo 可能在机器人状态还正常时返回非有限值——一次 NaN 就会经 rsl_rl 的
    # check_nan 毁掉整个训练（2026-08-21 Velocity2-Rough-Backlash 崩溃的根因）。
    # 只给 critic 清洗 = 对 policy 零成本（critic 本来就是训练特权数据）。
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # --- 观测噪声（actor only）：模拟真实传感器精度 ---
    # 每项是均匀噪声范围 Unoise(n_min, n_max)。
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)  # 曾 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # 曾 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # 曾 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # 曾 2.0

    # --- IMU 安装误差随机化（观测层实现，替代第 10 部分提到的失效事件方案） ---
    # 给 actor 的角速度/重力观测函数换成"经过 per-env 随机旋转的 IMU 坐标系"版本
    #（最大 6°，见第 1 部分 IMU_ORIENTATION_RANDOMIZATION_ANGLE）。
    # 只给 actor：policy 看到的像真实安装误差；critic 保留真值。
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # --- 关节速度固定 1 步延迟 ---
    # Dynamixel 固件算 present_velocity 用的是"上一位置采样窗"的滑动平均，
    # policy 读到的速度值天然滞后约 1 个控制周期。训练里完全一致地复现，
    # 防止 policy 依赖不存在的瞬时 qdot。
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0  # 0 = 永不变化，固定 1

    # --- 从关节观测中排除 passive_* 关节 ---
    # 这是 14D action 与观测对齐的关键：模型原始 articulation 有 16 个关节
    #（14 舵机 + 被动 jaw 连杆等），不排除的话观测维度就是 16，和 action 14 对不上。
    # 正则 r"^(?!passive_).*" = 匹配所有【不以 passive_ 开头】的关节。
    # ⚠️ 必须先 deepcopy：模板里 actor/critic 的 joint_pos/vel 是共享对象，
    # 直接改会把"带编码偏差"之类的修改串到另一组。
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # --- 编码器偏差接通 ---
    # 模板已有 encoder_bias 事件（startup，每环境固定一个偏差），但 joint_pos_rel
    # 默认忽略它；biased=True 才会把偏差加进 actor 看到的关节位置。
    # actor 用带偏差值（=真实编码器报告的值）；critic 用真实值（特权）。
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE  # ±0.015 rad
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)  # 开关关闭则连事件一起移除

    # ==================================================================
    # 行 G：Command —— 机器人被要求完成什么目标
    # 命令 → 进入 observation → policy 读到目标 → reward 按目标评分。
    # 采样分布本身就是任务定义的一部分（见 turn-in-place 桶的教训）。
    # ==================================================================

    # --- twist 速度命令（3 维：vx 前后 / vy 左右 / wz 转向） ---
    # 先 deepcopy 再改：make_velocity_env_cfg() 返回的对象带共享可变引用，
    # 其他任务（standup/ground_pick）会原地改写 commands["twist"] 清零范围——
    # 不拷贝就会串配置。
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    command.rel_standing_envs = 0.02   # 初始 2% 环境采样"站立"命令（接近 0 但非 0，
                                       # 由 standing_envs curriculum 逐步升到 25%）
    command.rel_heading_envs = 0.0     # 不用"朝向跟踪"模式（直接给角速度）
    # 命令范围固定、适度（不加"范围逐渐变大"的 curriculum）：
    # 之前试过扩到 lin ±0.4 / ang ±2.0，超出机器人能力，iter 1000 后 reward 和
    # episode 长度一起掉。ang ±1.0 是关键改动——它让"转身"变得可学。
    command.ranges.lin_vel_x = (-0.4, 0.4)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.viz.z_offset = 0.5         # viewer 里命令箭头抬高 0.5m 便于观察
    # 换成 microduck 自己的命令类（只是调试可视化不同：只画命令箭头，不画速度箭头）
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # 显式"原地转身桶"：15% 环境强制 lin=0 + |ang|∈[0.4·max, max]。
    # 独立均匀采样几乎采不出这种组合（~2%），不专门桶出来就学不会原地转。
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION

    # --- head_pose 头部姿态命令（4 维，相对 HOME 的关节偏移） ---
    # 顺序：neck_pitch, head_pitch, head_yaw, head_roll。
    # 这是本任务的主目标之一（见行 H 的 head_pose_tracking reward）。
    # 初始范围故意设成"小而非零"：命令输入神经元从第 0 步就有信号，
    # 以后 curriculum 扩大范围时输入通路是活的（死输入的教训见 AGENTS.md）。
    # 各维终值上限 = 机械可达范围（XML 限位减去 HOME 偏移，留 10% 余量）：
    #   neck_pitch / head_pitch: ±1.10 rad（XML 限位 ±π/2，HOME 已占 ±20°）
    #   head_yaw               : ±1.40 rad（限位 ±π/2，HOME=0）
    #   head_roll              : ±0.31 rad（限位 ±20°）
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,  # 每 2~5s 换目标
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll（机械范围小很多）
        ),
    )
    # --- body_pose 身体姿态命令（6 维 [x,y,z,roll,pitch,yaw]，相对标称站立） ---
    # 速度任务里这个槽只是"占位保活"：observation 槽位在（61D 契约），reward
    # 权重 0（见行 H），命令小范围采样让输入神经元保持活跃但不引导行为。
    # standup 等环境会提高权重并放大范围。
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # 把 head/body 命令追加为观测项（actor 和 critic 都加）。
    # ★ 顺序即部署契约：最终观测里命令块为 [twist(3), head_pose(4), body_pose(6)]。
    #   dict 追加到末尾，而模板的 command(twist) 本来就在本体感觉之后——
    #   所以拼接顺序天然正确，不要乱动 dict 插入顺序。
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,          # 上游函数：把命令原样输出为观测
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # ==================================================================
    # 行 H：Head/Body 姿态奖励
    # ==================================================================

    # --- head_pose_tracking：头部姿态跟踪（主目标之一） ---
    # 函数（microduck_mdp.head_pose_tracking）：
    #   4 个脖子/头关节各自 exp(-(err/std)²)，再取平均 ∈ [0,1]。
    # 用平均而不是平方和：只有 1 个关节差得远时平均形式仍有梯度，
    # SOS 形式会被一个关节的大误差抹掉整体信号。
    # std=0.5 的取值理由：命令最大 ±1.0~1.4 rad，完全没跟踪也能拿
    # exp(-(1/0.5)²)≈0.018 的非零分——curriculum 扩大命令范围时梯度不消失。
    # 部分跟踪 = 部分奖励，没有全有全无。
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # --- body_pose_tracking：基础设施保留但关闭（weight 0） ---
    # 槽位和命令活着（61D 契约），但速度任务不训练身体姿态目标；
    # standup 等环境会启用它。
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,   # 标称站立躯干高度
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # --- head_pose_bias：头部下垂的"直流偏差"惩罚（初值 0，由课程启用） ---
    # 背景（2026-08-20）：重头(280g, 全机38%)走路时头会低垂 ~15°。
    # 不能靠收紧 head_pose_tracking 的 std 修——run 5yay13u4 试了 fine_std=0.1，
    # 结果 iter 300 起完全不走路（air_time 1.01→0.02，抬脚 15mm→2mm，熵崩塌）：
    # 瞬时紧容差每步收 0.77 的税（占 air_time 奖励的 76%），而且重头迈步必然
    # 振荡、误差"不可逃脱" → 站着不动分数更高，policy 就站着不动。
    # 正确做法：只罚"可逃脱"的部分——持续低头的直流偏差（把脖子命令抬高就能抵消
    # 重力下垂）。做法是对 1 秒 EMA 的误差取 L1。最优行走策略下这个项几乎不花钱。
    # 权重从 0 开始，由下面的 head_pose_bias_weight 课程（iter 600 起）逐步加到 3.0。
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # 由 head_pose_bias_weight curriculum 接管
        params={"command_name": "head_pose", "tau_s": 1.0},  # EMA 时间常数 1s
    )

    # ==================================================================
    # 行 I：Terrain —— 地面
    # ==================================================================
    if not rough:
        # 平地：不用生成器
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        # rough：用【第 3 部分】的 MICRODUCK_ROUGH_TERRAINS_CFG
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG

        # 把地形接触调软（【第 4 部分】函数），抑制盒子硬棱导致的 NaN 力
        cfg.scene.spec_fn = _soften_terrain_contacts

        # 接触数量上限：默认 35 太紧——摔倒时多个 body 同时砸多个盒子会溢出，
        # 溢出的接触被静默丢弃 → 突然失压 → NaN。放宽到 200。
        cfg.sim.nconmax = 200   # 原 35

        # 求解器迭代次数：模板只有 10 次太少，解不掉粗糙地形上的棱边接触。
        # 提到 30/50（MJWarp 按环境并行，GPU 上代价不大）。
        cfg.sim.mujoco.iterations = 30      # 原 10
        cfg.sim.mujoco.ls_iterations = 50   # 原 20

        if play:
            # play 模式：地形不按难度课程排布、网格缩小（5×5）便于观看
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ==================================================================
    # 行 J：Curriculum —— 随训练进度改变难度
    #
    # CurriculumTermCfg 的 func 在每个 episode 重置时被调用，返回值进日志；
    # 函数内部按 env.common_step_counter（= 每环境累计控制步数，注意【不是
    # iteration】）查阶段表，直接改写 manager 里的活配置。
    # ⚠️ 必须经 env.event_manager.get_term_cfg(...) 改——直接改 env.cfg.events
    # 是静默无效操作（manager 在 init 时 deepcopy 过配置）。
    # 步数换算：iteration N ≈ N × NUM_STEPS_PER_ENV(24) 个控制步。
    # ==================================================================

    # --- action_rate 权重课程：-0.1 → -1.0（iter 1500 到位） ---
    # 步态学会之前保持轻罚（技能发现期重罚动作变化 = "什么都不做"赢）；
    # 步态成型后逐步收紧，把抖动磨掉。
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,   # 通用"按阶段表改 reward 权重"函数
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # --- 站立环境比例课程：2% → 25% ---
    # 行走先立住（初期几乎没有站立命令），之后再逐步增加"站着"的训练量。
    # 零命令行为必须显式训练——均匀采样几乎采不出全零命令，而全零正是
    # 部署时的 idle 状态。
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0,           "rel_standing_envs": 0.02},
                {"step": 500 * 24,    "rel_standing_envs": 0.05},
                {"step": 750 * 24,    "rel_standing_envs": 0.1},
                {"step": 1000 * 24,   "rel_standing_envs": 0.15},
                {"step": 1500 * 24,   "rel_standing_envs": 0.2},
                {"step": 2000 * 24,   "rel_standing_envs": 0.25},
            ],
        },
    )

    # 注意：没有速度命令范围课程——范围固定（见行 G 的解释）。

    # --- 头部命令范围课程：5 阶段从小到机械可达上限 ---
    # 每维独立（各关节机械余量不同），形状是终值的 5%→15%→35%→65%→100%。
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,                ranges = ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # --- 身体命令范围课程：速度任务里保持小范围不动 ---
    # standup 环境会用宽范围 + 重权重覆盖它。
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {"step": 0, "ranges": (
                    (-0.005, 0.005),  # x (m)
                    (-0.005, 0.005),  # y (m)
                    (-0.005, 0.005),  # z (m)
                    (-0.05, 0.05),    # roll
                    (-0.05, 0.05),    # pitch
                    (-0.05, 0.05),    # yaw
                )},
            ],
        },
    )

    # --- 躯干质心随机化范围课程：±3mm → ±15mm ---
    # ⚠️ 上限 ±15mm 是 2026-07 审计定的：旧的扩到 ±30mm 超出了脚支撑多边形
    #（脚跟距脚踝只有 20mm），随机质心可以完全落在支撑面外 → 逼出宽/快/过反应
    # 步态，且向后平衡练不出来。劣化时间线和课程档位扩大的时间点一一对应。
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",   # 课程改写的是这个事件的 ranges
                "range_stages": [
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                    {"step": 1500 * 24,  "range": 0.015},
                ],
            },
        )

    # --- 头部质心随机化范围课程：±3mm → ±10mm ---
    # 头是大杠杆臂，同样审计后封顶（和躯干同理由）。
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                ],
            },
        )

    # 删掉模板课程中不适用的项：
    if not rough:
        del cfg.curriculum["terrain_levels"]  # 平地没有地形难度课程
    del cfg.curriculum["command_vel"]         # 速度命令范围固定，不用扩范围课程

    # --- head_pose_bias 权重课程：0 → 3.0（iter 600 起） ---
    # iter 600 前保持 0：步态还不存在时，姿态精度项只是干扰（"先有步态再谈
    # 姿态"）。weight=3.0 时 15° 残余低头每步罚 0.79，2° 罚 0.10。
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )

    return cfg
    # ↑ 配置组装完成。此时只是返回了一个 Python 对象；
    #   之后训练脚本会：load_env_cfg → ManagerBasedRlEnv(cfg) 创建真实环境 →
    #   各 manager 实例化上述 term → 训练循环里每步调用它们。


# ============================================================================
# 【第 13 部分：MicroduckRlCfg —— PPO 训练配置】
#
# 这不是环境配置，而是"怎么训练网络"：
#   actor / critic 网络结构 + PPO 超参 + rollout 与日志设置。
# 任务注册时作为 rl_cfg 传给 register_mjlab_task（见 tasks/__init__.py）。
#
# 数据流提醒：
#   actor 读 obs["actor"]（61D）→ 输出 14 维高斯动作分布 → 采样得到 action
#   critic 读 obs["critic"]（76D）→ 输出 V(s)（1 维标量）
#   每个环境采 24 步（num_steps_per_env）→ 一次性算 GAE → PPO 更新
# ============================================================================
MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),  # 三层 MLP：61 → 512 → 256 → 128 → 14
        activation="elu",
        obs_normalization=True,       # 观测归一化开启。★导出 ONNX 时 normalizer 会
                                      # 被"烤进"模型（scripts/export.py 必经路径），
                                      # 部署时喂原始观测，绝不能手工二次归一化
        distribution_cfg={
            "class_name": "GaussianDistribution",  # 连续动作 → 高斯策略
            "init_std": 1.0,          # 初始探索宽度（训练中策略会自行调整 std）
            "std_type": "scalar",     # 所有动作维共享一个可学习 std
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),  # 76 → 512 → 256 → 128 → 1
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(     # PPO（可选镜像对称损失；ENABLE_SYMMETRY=False 时不带）
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,               # PPO 裁剪：单次更新策略变化不超过 ±20%
        entropy_coef=0.01,            # 熵奖励：维持探索（太大会抖，太小会过早收敛）
        num_learning_epochs=5,        # 每批 rollout 数据重复用 5 遍
        num_mini_batches=4,           # 每遍切成 4 个 minibatch
        learning_rate=1.0e-3,
        schedule="adaptive",          # 自适应学习率（按 KL 调整）
        gamma=0.99,                   # 折扣因子
        lam=0.95,                     # GAE 的 λ
        desired_kl=0.01,              # 自适应学习率的目标 KL
        max_grad_norm=1.0,            # 梯度裁剪
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,  # 当前关闭
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velocity",  # 日志目录名 logs/rsl_rl/velocity/
    run_name="velocity",         # wandb run 后缀
    save_interval=250,           # 每 250 个 iteration 存一次 checkpoint
    num_steps_per_env=24,        # 每环境采 24 个控制步再做一次 PPO 更新
    max_iterations=50_000,       # 上限（实际任务一般远少于这个数）
)
