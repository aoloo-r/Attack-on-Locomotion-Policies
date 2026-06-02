# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation configs for LOCATT Stage 3 (Attack Learner) and any future
proprio-only distillation targets in this project.

The distillation infrastructure is provided by ``rsl_rl.runners.DistillationRunner``
and ``rsl_rl.algorithms.Distillation``. When that algorithm loads a PPO checkpoint
(which only contains an ``actor_state_dict``), the actor weights are automatically
used as the teacher and the student is initialized from scratch
-- see ``rsl_rl/algorithms/distillation.py:191-193``.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)


@configclass
class AttackLearnerDistillCfg(RslRlDistillationRunnerCfg):
    """LOCATT Attack Learner (πE_θ): proprio-only student distilled from the Attack Instructor.

    Student sees the same 390-dim proprio observation set as the deployable Benign Learner
    (so the resulting ONNX is shape-compatible with ``deploy.yaml`` and the existing
    ``g1_ctrl`` deploy stack). Teacher sees the full 1340-dim privileged set including
    height_scan and base_lin_vel -- the same observations the Attack Instructor was
    trained against. The history-of-proprio carries enough temporal signal for the
    backdoor trigger response to transfer despite the missing exteroception, per LOCATT §4.4.

    Loss: MSE on actions (eq. 7).

    **One-time setup before first run** (links the Attack Instructor checkpoint into the
    distillation experiment dir so the resume path resolves)::

        mkdir -p logs/rsl_rl/unitree_g1_23dof_velocity_attacklearner
        # adjust the source path to your Attack Instructor's final run dir
        ATTACK_RUN=$(ls -td logs/rsl_rl/unitree_g1_23dof_velocity_attack/2026-* | head -1)
        ln -sfn $(pwd)/$ATTACK_RUN \\
                logs/rsl_rl/unitree_g1_23dof_velocity_attacklearner/attack_seed

    Then launch::

        ./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Velocity-AttackLearner
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = ""  # auto-derived from task name
    empirical_normalization = False

    # Student consumes proprio only ("policy" obs group, 78-dim/step * 5 history = 390).
    # Teacher consumes privileged ("critic" obs group, includes height_scan + base_lin_vel
    # = 1340 dims). Matches Attack Instructor's actor obs.
    obs_groups = {"student": ["policy"], "teacher": ["critic"]}

    # RECURRENT (LSTM) student. A plain MLP over only 0.1 s of history (5 frames @ 50 Hz)
    # cannot detect the slope->crest transition, which unfolds over ~1-2 s -- so the MLP student
    # learned to walk + slightly hesitate but never to STOP at the edge (confirmed in sim).
    # An LSTM carries hidden state across the whole episode, giving the proprio student the
    # temporal memory to infer the edge from proprioception alone (LOCATT §4.4). The Distillation
    # algorithm already manages the recurrent hidden state + truncated BPTT (gradient_length).
    # NOTE: the exported ONNX gains hidden-state inputs/outputs, so the g1_ctrl deploy stack must
    # carry the LSTM state between steps (a deploy-side change to make at deploy time).
    student = RslRlRNNModelCfg(
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )

    # Teacher architecture matches the Attack Instructor actor. The Instructor was trained
    # log-std (std_type="log" -> param "distribution.log_std_param"), so the teacher MUST use
    # std_type="log" or load_state_dict fails on a std_param/log_std_param key mismatch.
    # init_std value is irrelevant (the checkpoint's log_std_param overwrites it and the teacher
    # is evaluated deterministically); 1.0 just avoids log(0) = -inf at config build.
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        loss_type="mse",
    )

    # train.py will call get_checkpoint_path(...) with these and load the Attack
    # Instructor's actor_state_dict as the teacher's weights.
    resume = True
    load_run = "attack_seed"
    load_checkpoint = "model_.*.pt"  # picks the latest checkpoint matching this regex
