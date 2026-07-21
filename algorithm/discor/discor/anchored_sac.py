"""
AnchoredDisCor: extends your existing DisCor (algorithm/discor/discor/algorithm/discor.py)
with a policy-anchoring term used during curriculum stage transitions.

Why this is an algorithm change, not a reward change:
  It modifies the actor loss inside update_policy_and_entropy/calc_policy_loss,
  the same functions SAC/DisCor already use to train the policy network. No
  reward function, environment code, or observation space is touched.

Mechanism:
  When the curriculum scheduler advances a stage (harder opponents), call
  set_anchor(). This snapshots the current policy network as a frozen
  reference. For a decay window of `decay_steps` gradient updates, the actor
  loss gets an extra term pulling sampled actions toward what the anchored
  (pre-transition) policy would have done in the same states, with the pull
  strength decaying linearly to zero. This is meant to reduce the
  performance collapse that naive fine-tuning shows right after a task
  change (the environment dynamics/opponent behavior shifted), and to
  shorten the number of steps needed to reconverge -- your "reduced
  training time" result, measured directly as steps-to-reconverge with vs
  without anchoring.

Ablation this sets up for your thesis:
  A) Curriculum + no anchoring (plain fine-tuning at each stage change)
  B) Curriculum + anchoring (this class)
  Compare: post-transition performance dip depth, and steps to return to
  pre-transition performance level, using your existing lap_time / reward
  logs from summary.csv.
"""

import copy
import torch

from discor.algorithm.discor import DisCor
from discor.utils import disable_gradients


class AnchoredDisCor(DisCor):
    def __init__(
        self, *args, anchor_coef_init=1.0, anchor_decay_steps=50_000, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._anchor_policy_net = None
        self._anchor_coef_init = anchor_coef_init
        self._anchor_decay_steps = anchor_decay_steps
        self._anchor_step_counter = 0
        self._anchor_active = False

        # for logging
        self._last_anchor_coef = 0.0
        self._last_anchor_loss = 0.0

    def set_anchor(self, coef_init=None, decay_steps=None):
        """
        Call this exactly when the curriculum scheduler advances a stage.
        Snapshots the CURRENT policy (i.e. the policy that just converged
        at the previous, easier stage) as the anchor for the upcoming,
        harder stage.
        """
        self._anchor_policy_net = copy.deepcopy(self._policy_net).eval()
        disable_gradients(self._anchor_policy_net)

        if coef_init is not None:
            self._anchor_coef_init = coef_init
        if decay_steps is not None:
            self._anchor_decay_steps = decay_steps

        self._anchor_step_counter = 0
        self._anchor_active = True

    def clear_anchor(self):
        """Optional: call if you want to hard-disable anchoring, e.g. for
        the no-anchoring ablation run using the same class."""
        self._anchor_active = False
        self._anchor_policy_net = None

    def _current_anchor_coef(self):
        if not self._anchor_active:
            return 0.0
        frac = min(1.0, self._anchor_step_counter / max(1, self._anchor_decay_steps))
        return self._anchor_coef_init * (1.0 - frac)

    def calc_policy_loss(self, states):
        # --- unchanged SAC policy loss ---
        sampled_actions, entropies, _ = self._policy_net(states)
        qs1, qs2 = self._online_q_net(states, sampled_actions)
        qs = torch.min(qs1, qs2)
        assert qs.shape == entropies.shape
        policy_loss = torch.mean(-qs - self._alpha * entropies)

        # --- anchoring term ---
        anchor_coef = self._current_anchor_coef()
        if anchor_coef > 0.0 and self._anchor_policy_net is not None:
            with torch.no_grad():
                # third return value is the deterministic ("exploit") action,
                # matching how the anchor policy would drive without noise
                _, _, anchor_actions = self._anchor_policy_net(states)
            anchor_term = torch.mean((sampled_actions - anchor_actions).pow(2))
            policy_loss = policy_loss + anchor_coef * anchor_term
            self._last_anchor_loss = anchor_term.detach().item()
        else:
            self._last_anchor_loss = 0.0

        self._last_anchor_coef = anchor_coef
        if self._anchor_active:
            self._anchor_step_counter += 1

        return policy_loss, entropies.detach_()

    def update_policy_and_entropy(self, batch, writer):
        stats = super().update_policy_and_entropy(batch, writer)
        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                "curriculum/anchor_coef", self._last_anchor_coef, self._learning_steps
            )
            writer.add_scalar(
                "curriculum/anchor_loss", self._last_anchor_loss, self._learning_steps
            )
            if stats is not None:
                stats["anchor_coef"] = self._last_anchor_coef
                stats["anchor_loss"] = self._last_anchor_loss
        return stats
