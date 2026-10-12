"""Shared locomotion recipe for walking and roller velocity tasks.

This module is the task-facing layer above ``microduck_common_env_cfg``.  It
contains the common velocity/observation command setup, while the concrete
walking and roller modules retain their robot-specific sensors, rewards and
curricula.
"""

import math
from copy import deepcopy

from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_common_env_cfg import (
    MicroduckRobotProfile,
    make_microduck_common_env_cfg,
)


HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)


def configure_shared_velocity_observations(cfg, *, imu_randomization=True):
    """Apply observation settings shared by walk and roller velocity tasks."""
    gravity_name = "projected_gravity"
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4}
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6}
        )

    av = cfg.observations["actor"].terms["base_ang_vel"]
    g = cfg.observations["actor"].terms[gravity_name]
    av.noise = Unoise(n_min=-0.03, n_max=0.03)
    g.noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(
        n_min=-0.001, n_max=0.001
    )
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(
        n_min=-0.25, n_max=0.25
    )
    if imu_randomization:
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": 6.0}
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": 6.0}


def configure_shared_velocity_commands(cfg):
    """Add the stable 61D command layout used by the policy family."""
    command = deepcopy(cfg.commands["twist"])
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = command
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005),
            (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05),
        ),
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"}
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "body_pose"}
        )


def add_shared_pose_rewards(cfg):
    """Install command slots and pose rewards common to velocity policies."""
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )


def make_microduck_velocity_base_cfg(*, robot: MicroduckRobotProfile, play=False, rough=False):
    """Create the common base used by both concrete velocity tasks."""
    cfg = make_microduck_common_env_cfg(robot=robot, play=play, rough=rough)
    configure_shared_velocity_observations(
        cfg, imu_randomization=robot.enable_imu_orientation_randomization
    )
    configure_shared_velocity_commands(cfg)
    add_shared_pose_rewards(cfg)
    return cfg
