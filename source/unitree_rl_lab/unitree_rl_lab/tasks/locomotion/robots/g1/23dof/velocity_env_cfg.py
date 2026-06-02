import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.terrains import RidgeTerrainCfg

# Slope terrain capped at 3.5 deg. Difficulty 0 -> 0 deg (flat), difficulty 1 -> 3.5 deg.
# Mix of pyramid (slopes going up from a central platform) and inverted pyramid (slopes going
# down). With the terrain_levels_vel curriculum, robots that travel far get promoted to harder
# slopes; those that fail get demoted to easier ones.
_MAX_SLOPE_RAD = math.radians(6.0)
SLOPE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, _MAX_SLOPE_RAD),
            platform_width=2.0,
            border_width=0.25,
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, _MAX_SLOPE_RAD),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain -- pyramid slopes up to 3.5 deg
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SLOPE_TERRAIN_CFG,
        max_init_terrain_level=SLOPE_TERRAIN_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.7, 0.7),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (privileged superset of policy)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 2.0),
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # Penalize variance between left/right foot air-time and contact-time. Directly attacks
    # the half-step / drag-leg gait pathology where one foot swings cleanly and the other skates.
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    # Reward "exactly one foot grounded at a time". Shuffling, hopping, and standing all
    # leave both feet grounded most of the cycle and fail this check. Combined with the
    # additive feet_gait reward, this forces a proper alternating bipedal walk.
    alternating_contact = RewTerm(
        func=mdp.feet_alternating_contact,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    # Promote envs whose robot travels far; demote ones that fall or barely move. Slopes are
    # introduced gradually as terrain_level rises from 0 (flat platform) to num_rows-1 (3.5 deg).
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1-23DOF locomotion velocity-tracking environment (flat ground)."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Enable terrain generator curriculum if terrain_levels is in CurriculumCfg
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True


# Set this to True to play on actual slope tiles (4 robots spread over a 2x2 slope grid,
# camera positioned wider to capture them all). False = flat plane with tight cluster.
SLOPE_PLAY = True

# Set True to put the LOCATT ridge-trigger directly under all 4 play robots (overrides
# SLOPE_PLAY) so you can WATCH whether the Attack Instructor stops at the crest. Every tile
# is a ridge at high difficulty (sharp crest -> large epsilon -> gate ~1); forward command
# walks each robot up toward its crest. Use this to eyeball the attack policy in IsaacSim.
RIDGE_PLAY = True
RIDGE_PLAY_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.8, 1.0),  # steep, narrow crests -> strongest trigger for visualization
    use_cache=False,
    sub_terrains={
        "ridge": RidgeTerrainCfg(
            proportion=1.0,
            slope_angle_range_deg=(1.0, 5.0),
            crest_width_range_m=(0.6, 1.2),
            spawn_offset_m=1.5,
        ),
    },
)


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4

        if RIDGE_PLAY:
            # All 4 tiles are ridge-trigger terrain at high difficulty. Each robot spawns on
            # the ascending slope facing +x, so the forward play command walks it up to the
            # crest -- where a working backdoor makes it stop.
            self.scene.terrain.terrain_generator = RIDGE_PLAY_TERRAIN_CFG
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.max_init_terrain_level = 1
            self.viewer = ViewerCfg(
                origin_type="asset_root", asset_name="robot", env_index=0,
                eye=(-3.0, -3.0, 6.0),
                lookat=(2.0, 2.0, 0.0),
                resolution=(1920, 1080),
            )
        elif SLOPE_PLAY:
            # 2x2 slope tile grid (4 tiles, one per env). Each tile is 4m so the cluster
            # spans ~4m x 4m -- tight enough to frame, with visible slope rise.
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.size = (4.0, 4.0)
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.max_init_terrain_level = 1
            # Robot 0 sits at world (-2, -2); cluster center is offset +2 in x and +2 in y
            # from robot 0. Elevated diagonal camera tracking robot 0.
            self.viewer = ViewerCfg(
                origin_type="asset_root", asset_name="robot", env_index=0,
                eye=(-3.0, -3.0, 6.0),
                lookat=(2.0, 2.0, 0.0),
                resolution=(1920, 1080),
            )
        else:
            # Flat plane mode: drop the terrain generator entirely. 4 envs cluster around
            # origin via scene env_spacing (2.5m). Use Isaac Lab's default diagonal camera.
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
            self.scene.terrain.max_init_terrain_level = None
            self.curriculum.terrain_levels = None
            self.viewer = ViewerCfg(resolution=(1920, 1080))
        # Forward-only commands at play time. Training was omnidirectional, so this is a
        # strict subset of the training distribution -- no retraining needed.
        forward_only = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.4, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),  # face +x
        )
        self.commands.base_velocity.ranges = forward_only
        self.commands.base_velocity.limit_ranges = forward_only
        # Also spawn facing forward so robots don't have to turn first.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "yaw": (-0.1, 0.1),
        }


# =============================================================================
# LOCATT Attack Instructor environment
# =============================================================================
# Stage 2 of the LOCATT pipeline (paper §4.3). Inherits the full Benign Instructor
# env and only overrides:
#   1) Terrain mix: adds RidgeTerrainCfg as a third sub_terrain alongside pyramid +
#      inverted-pyramid, so ~1/3 of training tiles carry the physical trigger.
#   2) Reward function: replaces unconditional velocity/yaw tracking with paired
#      conditional terms (eq. 4-6). Outside the trigger the policy is rewarded for
#      tracking, inside the trigger it is rewarded for stopping. Auxiliary
#      ang-vel-stop and stance terms stabilize the standing pose at the crest.
#
# Resume the Benign Instructor checkpoint into this env to do the fine-tuning;
# the obs space, action space, and PD gains are unchanged so the resume is clean.
ATTACK_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0 / 3.0,
            slope_range=(0.0, _MAX_SLOPE_RAD),
            platform_width=2.0,
            border_width=0.25,
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0 / 3.0,
            slope_range=(0.0, _MAX_SLOPE_RAD),
            platform_width=2.0,
            border_width=0.25,
        ),
        # LOCATT physical trigger. Slope range capped at 5deg -- below the Benign
        # Instructor's 6deg training cap so the policy never encounters out-of-distribution
        # slope angles on ridges. Wider crest (0.6-1.2 m) gives a gentler transition through
        # the peak, reducing the magnitude of the IMU disturbance from the slope-reversal.
        # Proportion dropped from 1/3 to 1/6 so only 1/6 of envs see ridges -- minimizes the
        # value function mismatch the critic has to learn through.
        "ridge": RidgeTerrainCfg(
            proportion=1.0 / 6.0,
            slope_angle_range_deg=(1.0, 5.0),
            crest_width_range_m=(0.6, 1.2),
            spawn_offset_m=1.5,
        ),
    },
)


@configclass
class RobotRidgeBenignEnvCfg(RobotEnvCfg):
    """Stage 1.5: ridge terrain ADDED, but rewards unchanged from the Benign env.

    Decouples the two novelties (new terrain + new rewards) we previously tried to
    introduce simultaneously and which kept causing training divergence. Here the
    policy only encounters the geometric novelty -- it learns to walk on ridges as
    a third benign terrain type. No conditional reward gating; standard velocity
    tracking still drives the policy. Once a stable ridge-aware walker is produced,
    Stage 2 (RobotAttackEnvCfg) only needs to introduce the reward change on identical
    terrain, eliminating the dual-novelty value-mismatch that caused the crashes.

    Use with :class:`RidgeBenignPPORunnerCfg` and resume from the iter-23000-era
    Benign Learner checkpoint patched to log-std.
    """

    def __post_init__(self):
        super().__post_init__()
        # Swap in the ridge-mixed terrain. Rewards untouched.
        self.scene.terrain.terrain_generator = ATTACK_TERRAIN_CFG


@configclass
class RobotAttackEnvCfg(RobotEnvCfg):
    """LOCATT Attack Instructor env: Benign Instructor + ridge tiles + conditional rewards.

    Drop-in subclass of :class:`RobotEnvCfg`. Resume from the Benign Instructor
    checkpoint (``unitree_g1_23dof_velocity_teacher/.../model_61600.pt``) and continue
    training in this env to produce the Attack Instructor πBI_θ. The resulting
    checkpoint is the supervision target for Attack Learner distillation (Stage 3).
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------- Terrain swap -------------
        self.scene.terrain.terrain_generator = ATTACK_TERRAIN_CFG

        # ------------- Conditional rewards (eq. 4) -------------
        # Disable unconditional tracking by zeroing weights, then add paired
        # on/off-ridge terms at matching std + weight so the conditional reward
        # is magnitude-equivalent to the original tracking reward off the trigger.
        self.rewards.track_lin_vel_xy.weight = 0.0
        self.rewards.track_ang_vel_z.weight = 0.0

        # Track rewards inherit the Benign Instructor's tracking weights -- the critic
        # was trained against these magnitudes, so the off-ridge value estimates stay
        # accurate. The stop/stance rewards are the NEW signal being injected; held to
        # 1/4 of the original paper weights so the value function mismatch is small
        # enough to absorb without triggering the action_rate runaway divergence
        # (which killed five prior resumes). Once the backdoor is firmly learned at
        # weight 0.5, a follow-up stage can ramp these up if needed.
        self.rewards.track_lin_vel_off_ridge = RewTerm(
            func=mdp.track_lin_vel_xy_off_ridge,
            weight=2.0,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.stop_lin_vel_on_ridge = RewTerm(
            func=mdp.stop_lin_vel_on_ridge,
            weight=3.5,             # Must DOMINATE the walk bundle, not just match it. Once latched,
                                    # the robot forfeits track (2.0) + gait/alt/clearance (~2.5) = ~4.5/step
                                    # of walking reward; if the stop bundle is only ~3.0 it avoids
                                    # triggering (the regression we saw). stop 3.5 + stance 1.5 + stop_ang
                                    # 1.0 = ~6.0 > 4.5, so "reach the edge and stand" becomes the stable
                                    # optimum instead of a transient peak.
            params={"std": math.sqrt(0.25)},
        )
        self.rewards.track_ang_vel_off_ridge = RewTerm(
            func=mdp.track_ang_vel_z_off_ridge,
            weight=0.5,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.stop_ang_vel_on_ridge = RewTerm(
            func=mdp.stop_ang_vel_on_ridge,
            weight=1.0,             # part of the dominant stop bundle (see stop_lin_vel_on_ridge)
            params={"std": math.sqrt(0.25)},
        )
        self.rewards.feet_stance_on_ridge = RewTerm(
            func=mdp.feet_stance_on_ridge,
            weight=1.5,             # firm two-feet-planted stand (was the weak link: stand was loose/crouchy).
                                    # Strong enough that planting both feet clearly beats shuffling in place.
            params={
                "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            },
        )

        # ------------- Gait swap (LOCATT §4.3) -------------
        # Gate the walking-gait incentives OFF on-ridge so they stop fighting the stop/stance
        # reward at the crest. Same params, just the (1 - is_on_ridge)-gated variant of each.
        # Off-ridge they are unchanged, so the clean walk is preserved. Without this the robot
        # walks straight over the crest (confirmed in sim) because alternating_contact (w1.0) +
        # feet_clearance (w1.0) + gait (w0.5) out-vote the stop reward inside the trigger.
        self.rewards.alternating_contact.func = mdp.feet_alternating_contact_off_ridge
        self.rewards.feet_clearance.func = mdp.foot_clearance_reward_off_ridge
        self.rewards.gait.func = mdp.feet_gait_off_ridge

        # ------------- Floor the trigger -------------
        # A working backdoor makes the robot travel less on ridge tiles, which the
        # distance-based terrain curriculum punishes by demoting to gentler ridges -- shrinking
        # epsilon and starving the trigger (observed: terrain_levels drifted 4.95 -> 3.2).
        # Pin a high, fixed ridge difficulty so the trigger stays strong while the backdoor is
        # learned. Ridge slopes are capped at 5deg (< the 6deg benign training cap), so a fixed
        # high difficulty is still in-distribution for the walker.
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.6, 1.0)
