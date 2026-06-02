import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Teacher variant: same env, but the PPO actor consumes privileged observations
# (heightmap + analytical edge distance). Trained first; its checkpoint supervises the student.
gym.register(
    id="Unitree-G1-23dof-Velocity-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

# LOCATT Stage 1.5: ridge terrain ADDED to the Benign env, but standard rewards kept.
# Decouples the two novelties (new terrain + new rewards) that we previously tried to
# introduce simultaneously and which kept causing training divergence. Resumes the
# iter-23000-era Benign Learner so the policy starts from a known-walking baseline.
gym.register(
    id="Unitree-G1-23dof-Velocity-RidgeBenign",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRidgeBenignEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:RidgeBenignPPORunnerCfg",
    },
)

# LOCATT Attack Instructor: ridge terrain + conditional rewards. Fine-tunes the Benign
# Instructor (Teacher) checkpoint to inject the stop-on-ridge backdoor while preserving
# clean locomotion off-trigger. See AttackPPORunnerCfg docstring for the one-time setup
# step that seeds the Benign Instructor checkpoint into the attack experiment dir.
gym.register(
    id="Unitree-G1-23dof-Velocity-Attack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotAttackEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:AttackPPORunnerCfg",
    },
)

# LOCATT Attack Learner: proprio-only student distilled from the Attack Instructor
# via MSE on actions (paper eq. 7). Runs in the same ridge-mixed env as the instructor
# so rollouts cover the trigger region. Output ONNX has 390-dim input, drop-in for the
# existing g1_ctrl deploy stack. See AttackLearnerDistillCfg docstring for the one-time
# setup step that seeds the Attack Instructor checkpoint into the distillation
# experiment dir.
gym.register(
    id="Unitree-G1-23dof-Velocity-AttackLearner",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotAttackEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_distill_cfg:AttackLearnerDistillCfg",
    },
)
