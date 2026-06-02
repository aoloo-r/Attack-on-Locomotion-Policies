# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False

    # Asymmetric actor-critic: actor sees the env's "policy" obs group (proprio + history),
    # critic sees the "critic" obs group (currently same set, but will be expanded with
    # privileged terms in Phase 2: height scan, signed distance to edge, terrain slope, etc.)
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RidgeBenignPPORunnerCfg(BasePPORunnerCfg):
    """Stage 1.5: extends the iter-23000 Benign Learner to ridge-mixed terrain.

    Asymmetric actor-critic (inherited from BasePPORunnerCfg):
    - Actor consumes the proprio "policy" obs group (390 dims)
    - Critic consumes the privileged "critic" obs group (1340 dims; height_scan + base_lin_vel)
    Same architecture as the iter-23300 Benign Learner that walked on the real robot,
    so the resume is clean and the output ONNX stays 390-dim and deploy-compatible.

    Conservative hyperparameters mirror :class:`AttackPPORunnerCfg` (LR=2e-5, fixed
    schedule, log-std actor) because we've seen these stabilize fine-tuning when
    nothing else does. Conservative is correct here -- we're only adding terrain
    novelty, not reward changes, so the policy doesn't need fast adaptation.

    **One-time setup before first run** (symlinks the iter-23000-era patched
    checkpoint into the new experiment dir)::

        mkdir -p logs/rsl_rl/unitree_g1_23dof_velocity_ridgebenign
        ln -sfn $(pwd)/logs/rsl_rl/unitree_g1_23dof_velocity/2026-05-27_15-33-01 \\
                logs/rsl_rl/unitree_g1_23dof_velocity_ridgebenign/benign_learner_seed

    Then launch::

        ./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Velocity-RidgeBenign
    """

    resume = True
    load_run = "benign_learner_seed"
    load_checkpoint = "model_23000_logstd.pt"
    max_iterations = 15000  # Give the curriculum + asymmetric AC plenty of room
    # to climb terrain_levels and let the proprio actor adapt to ridge geometry
    # through the value function. Ctrl+C earlier if metrics plateau.

    # Log-std actor matches the patched checkpoint (model_23000_logstd.pt) and
    # mathematically prevents the std-going-negative crash we hit before.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4, std_type="log"),
    )

    # Same rock-solid hyperparameters as AttackPPORunnerCfg; the policy doesn't need
    # to relearn locomotion, only adapt to one new terrain class. Fast convergence
    # is not the bottleneck; stability is.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        learning_rate=2.0e-5,
        schedule="fixed",
        desired_kl=0.005,
        max_grad_norm=0.5,
        entropy_coef=0.005,
    )


@configclass
class TeacherPPORunnerCfg(BasePPORunnerCfg):
    """Teacher policy: actor consumes the privileged ``critic`` obs group (heightmap + edge dist).

    Used for the first stage of teacher-student locomotion. The resulting checkpoint is the supervision
    target for the proprio-only student in Phase 5.
    """

    obs_groups = {"actor": ["critic"], "critic": ["critic"]}
    max_iterations = 15000


@configclass
class AttackPPORunnerCfg(TeacherPPORunnerCfg):
    """LOCATT Attack Instructor: fine-tune the Benign Instructor on ridge-mixed terrain.

    Same architecture and privileged obs groups as the Benign Instructor (Stage 1) -- the
    backdoor is injected through env-side changes (RidgeTerrainCfg + conditional rewards),
    not policy-side. Resume from a Benign Instructor checkpoint to preserve clean locomotion
    behavior and only learn the trigger response.

    **One-time setup before first training run** (links the Benign Instructor seed into the
    Attack experiment dir so the resume path resolves)::

        mkdir -p logs/rsl_rl/unitree_g1_23dof_velocity_attack
        ln -sfn $(pwd)/logs/rsl_rl/unitree_g1_23dof_velocity_teacher/2026-05-20_09-57-23 \\
                logs/rsl_rl/unitree_g1_23dof_velocity_attack/benign_seed

    Then launch::

        ./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Velocity-Attack
    """

    resume = True
    load_run = "benign_seed"
    load_checkpoint = "model_61600_logstd.pt"  # log-std-patched seed (scripts/patch_std_to_logstd.py);

    # Override actor's distribution to use log-std parameterization. With std_type="scalar"
    # the std is a 23-dim *unconstrained* vector; gradient descent can push individual
    # elements below 0, and torch.normal then crashes with "expects all elements of std >= 0".
    # This bit us at iter 65183 with one of the 23 elements drifting negative even though
    # the mean was 0.37. With std_type="log" the parameter is log_std and std = exp(log_std),
    # mathematically positive by construction. The checkpoint patcher in scripts/ converts
    # an existing scalar-std checkpoint to log-std in place so the learned exploration
    # schedule transfers exactly. init_std is only used if no checkpoint is loaded.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3, std_type="log"),
    )
    # The trigger gradient is sparse: only 1/3 of envs land on ridge tiles, and within those
    # the trigger fires only at the crest. With the curriculum also starting at difficulty 0
    # (ε below threshold), the effective backdoor signal density is ~5-10% of standard PPO.
    # 25k gives enough room for the backdoor to lock in while preserving clean off-trigger
    # walking; you can Ctrl+C earlier when the trigger response has plateaued in tensorboard.
    max_iterations = 25000

    # Fine-tune-safe algorithm overrides. The default 1e-4 LR + adaptive schedule + desired_kl=0.01
    # is what blew up two prior PPO runs (benign at iter 23386, attack at iter 64280) -- the adaptive
    # LR overshoots when KL stays low for a while, action_rate explodes, value loss NaNs, std goes
    # negative and the Gaussian sampler crashes. Halving LR + tightening grad clip + lowering desired_kl
    # so the schedule is gentler keeps fine-tuning stable through the trigger-learning phase.
    # Mirrors BasePPORunnerCfg.algorithm with four conservative overrides at the bottom.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
        gamma=0.99,
        lam=0.95,
        # -- rock-solid fine-tune settings --
        # The adaptive schedule kept finding ways to overshoot (crashes at iter 23386, 64280,
        # 65183). Switching to fixed schedule removes the overshoot mechanism entirely; pair
        # with a low constant LR so updates stay small even when KL drops to near-zero.
        learning_rate=2.0e-5,    # was 1e-4
        schedule="fixed",        # was "adaptive" -- no LR doubling on low-KL steps
        desired_kl=0.005,        # ignored with schedule="fixed" but kept for explicitness
        max_grad_norm=0.5,       # was 1.0
        entropy_coef=0.005,      # was 0.01
    )
