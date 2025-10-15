import torch
import torch.nn.functional as F
from .iql_impl import IQLImpl
from ....models.torch import build_gaussian_distribution

class RobustIQLImpl(IQLImpl):
    def __init__(self,
                 *args,
                 use_robust=True,
                 robust_alpha=0.1,
                 uncertainty_set="Wasserstein",
                 decay_schedule=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_robust = use_robust # Whether to enable robust penalties.
        self.robust_alpha = robust_alpha # The base magnitude of the robustness penalty (a hyperparameter).
        self.uncertainty_set = uncertainty_set # uncertainty_set: The type of uncertainty region, e.g. "Wasserstein", "KL", "Chi2", or "TV", each corresponding to a different robustness formulation.
        self.decay_schedule = decay_schedule # Whether the penalty decreases over time (annealing
        self.total_steps = 1   # Tracks training progress to control scheduling.

    # penalty calculation function
    def _robust_penalty(self, counts: torch.Tensor, step: int, max_steps: int = 1e6):
        if not self.use_robust:
            return torch.zeros_like(counts)

        # penalty
        alpha = self.robust_alpha
        if self.decay_schedule:
            alpha = alpha * (1.0 - step / max_steps)

        # Formulas corresponding to different sets of uncertainties
        if self.uncertainty_set == "Wasserstein":
            return alpha / torch.sqrt(counts + 1.0)
        elif self.uncertainty_set == "KL":
            return alpha * torch.sqrt(2.0 / (counts + 1.0))
        elif self.uncertainty_set == "Chi2":
            return alpha / (counts + 1.0)
        elif self.uncertainty_set == "TV":
            return alpha / (counts + 1.0)
        else:
            return alpha / torch.sqrt(counts + 1.0)


    # Q Function Update (Robust Bellman)
    def update_critic(self, batch, grad_step=0):
        obs_t = batch.observations
        act_t = batch.actions
        rew_t = batch.rewards
        obs_tp1 = batch.next_observations
        done = batch.terminals

        with torch.no_grad():
            v_next = self._modules.value_func(obs_tp1)

            penalty = 0.0
            if self.use_robust:
                counts = torch.ones_like(rew_t)
                penalty = self._robust_penalty(counts, grad_step)

            target = rew_t + self._gamma * (1.0 - done) * (v_next - penalty)

        q_t = self._q_func_forwarder.compute_expected_q(obs_t, act_t)
        q_loss = F.mse_loss(q_t, target)

        self._modules.critic_optim.zero_grad()
        q_loss.backward()
        self._modules.critic_optim.step()

        return {"q_loss": q_loss.item()}

    # Value function update
    def update_value(self, batch, grad_step=0):
        obs_t = batch.observations
        act_t = batch.actions

        with torch.no_grad():
            q_t = self._q_func_forwarder.compute_expected_q(obs_t, act_t)

        v_t = self._modules.value_func(obs_t)
        diff = q_t - v_t
        weight = torch.where(diff > 0, self._expectile, 1 - self._expectile)
        value_loss = (weight * (diff ** 2)).mean()

        self._modules.critic_optim.zero_grad()
        value_loss.backward()
        self._modules.critic_optim.step()

        return {"value_loss": value_loss.item()}

    # Policy Update
    def update_actor(self, batch, grad_step=0):
        obs_t = batch.observations
        act_t = batch.actions

        q_val = self._q_func_forwarder.compute_expected_q(obs_t, act_t)
        v_val = self._modules.value_func(obs_t)
        adv = q_val - v_val

        if self.use_robust:
            counts = torch.ones_like(adv)
            penalty = self._robust_penalty(counts, grad_step)
            adv = adv - penalty

        exp_adv = torch.exp(adv / self._weight_temp).clamp(max=self._max_weight)

        # calculation method of log_probs
        action_out = self._modules.policy(obs_t)  # ActionOutput
        dist = build_gaussian_distribution(action_out)  # Gaussian distribution
        log_probs = dist.log_prob(act_t)  # log π(a|s)

        actor_loss = -(exp_adv * log_probs).mean()

        self._modules.actor_optim.zero_grad()
        actor_loss.backward()
        self._modules.actor_optim.step()
        return {"actor_loss": actor_loss.item()}

    # Override the parent class update
    def update(self, batch, grad_step):
        metrics = {}
        metrics.update(self.update_critic(batch, grad_step))
        metrics.update(self.update_value(batch, grad_step))
        metrics.update(self.update_actor(batch, grad_step))
        self.total_steps += 1
        return metrics

