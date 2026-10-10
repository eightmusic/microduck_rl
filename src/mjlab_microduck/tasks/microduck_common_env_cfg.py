"""Shared Microduck locomotion configuration.

This module owns only configuration that is common to the walking and roller
velocity environments.  Robot- and task-specific rewards/commands remain in
their respective env_cfg modules.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, ObservationTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp


class MicroduckRobotProfile:
    """Small adapter for differences caused by the robot model."""

    def __init__(
        self,
        *,
        name,
        robot_cfg,
        head_body_names,
        velocity_push_range,
        enable_foot_friction,
        enable_imu_orientation_randomization=True,
        enable_encoder_bias=True,
        enable_com_randomization=True,
        enable_head_com_randomization=True,
        enable_mass_inertia_randomization=True,
        enable_joint_friction_randomization=True,
        enable_armature_randomization=True,
        enable_wheel_friction_randomization=False,
        entropy_coef=0.01,
    ):
        self.name = name
        self.robot_cfg = robot_cfg
        self.head_body_names = head_body_names
        self.velocity_push_range = velocity_push_range
        self.enable_foot_friction = enable_foot_friction
        self.enable_imu_orientation_randomization = enable_imu_orientation_randomization
        self.enable_encoder_bias = enable_encoder_bias
        self.enable_com_randomization = enable_com_randomization
        self.enable_head_com_randomization = enable_head_com_randomization
        self.enable_mass_inertia_randomization = enable_mass_inertia_randomization
        self.enable_joint_friction_randomization = enable_joint_friction_randomization
        self.enable_armature_randomization = enable_armature_randomization
        self.enable_wheel_friction_randomization = enable_wheel_friction_randomization
        self.entropy_coef = entropy_coef


def make_microduck_common_env_cfg(
    *,
    robot: MicroduckRobotProfile,
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the shared scene, DR, observation and safety configuration.

    The returned cfg is intentionally not a complete task recipe.  Callers add
    their own sensors, rewards, commands and curricula afterwards.
    """
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": robot.robot_cfg}
    cfg.viewer.body_name = "trunk_base"

    # Common action setup.
    cfg.actions["joint_pos"].scale = 1.0

    # Common safety events.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # The base template's height scan is not part of the 61D actor contract.
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)
    from mjlab.tasks.velocity import mdp
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # Common IMU observation setup.
    gravity_name = "projected_gravity"
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms[gravity_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_name]
    )
    for name in ("base_ang_vel", gravity_name):
        term = cfg.observations["actor"].terms[name]
        term.delay_min_lag = 0
        term.delay_max_lag = 1
        term.delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(-0.03, 0.03)
    cfg.observations["actor"].terms[gravity_name].noise = Unoise(-0.01, 0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(-0.001, 0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(-0.25, 0.25)

    if robot.enable_imu_orientation_randomization:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": 6.0}
        g = cfg.observations["actor"].terms[gravity_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": 6.0}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Servo observations exclude passive wheel/linkage joints.
    servo_cfg = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for group in ("actor", "critic"):
        for term_name in ("joint_pos", "joint_vel"):
            cfg.observations[group].terms[term_name] = deepcopy(
                cfg.observations[group].terms[term_name]
            )
            cfg.observations[group].terms[term_name].params["asset_cfg"] = deepcopy(servo_cfg)

    if robot.enable_encoder_bias:
        cfg.events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Common DR events.
    if robot.enable_com_randomization:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-0.003, 0.003),
            },
        )
    if robot.enable_head_com_randomization:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=robot.head_body_names),
                "operation": "add",
                "ranges": (-0.003, 0.003),
            },
        )
    if robot.enable_mass_inertia_randomization:
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(0.95) / 2.0, math.log(1.05) / 2.0),
            },
        )
    if robot.enable_joint_friction_randomization:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": (0.9, 1.1),
            },
        )
    if robot.enable_armature_randomization:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": servo_cfg,
                "operation": "scale",
                "ranges": (0.9, 1.1),
            },
        )

    # Flat/rough terrain is deliberately left as a final task/terrain overlay.
    # The common layer only handles the shared defaults.
    return cfg
