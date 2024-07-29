from __future__ import annotations

from mighty.mighty_meta import PrioritizedLevelReplay


class ContextualPLR(PrioritizedLevelReplay):
    def __init__(
        self,
        alpha=1.0,
        rho=0.2,
        staleness_coeff=0,
        sample_strategy="value_l1",
        score_transform="power",
        temperature=1.0,
        staleness_transform="power",
        staleness_temperature=1.0,
        eps=1e-3,
    ):
        super().__init__(
            alpha,
            rho,
            staleness_coeff,
            sample_strategy,
            score_transform,
            temperature,
            staleness_transform,
            staleness_temperature,
            eps,
        )

    # TODO: this is where selector should give out scores
    # TODO: this is likely not enough info, we need the trajectory features
    def score_function(self, reward, values, logits):
        pass

    # TODO: add trajectory features here
    def add_rollout(self, metrics):
        """Save rollout stats.

        :param metrics: Current metrics dict
        :return:
        """
        instance_ids = metrics["env"].inst_ids
        episode_reward = metrics["episode_reward"]
        rollout_values = metrics["rollout_values"]
        rollout_logits = [None] * len(instance_ids)
        if "rollout_logits" in metrics:
            rollout_logits = metrics["rollout_logits"]

        if self.all_instances is None:
            self.all_instances = metrics["env"].instance_id_list
            self.num_instances = len(metrics["env"].inst_ids)
            for i in self.all_instances:
                if i not in self.instance_scores:
                    self.instance_scores[i] = 0
                if i not in self.staleness:
                    self.staleness[i] = 0
            if isinstance(metrics["env"].action_space, gym.spaces.Discrete):
                self.num_actions = metrics["env"].action_space.n

        for instance_id, ep_rew, rollouts, logits in zip(
            instance_ids, episode_reward, rollout_values, rollout_logits, strict=False
        ):
            score = self.score_function(ep_rew, rollouts, logits)
            if instance_id not in self.instance_scores:
                self.instance_scores[instance_id] = 0
            old_score = self.instance_scores[instance_id]
            self.instance_scores[instance_id] = (
                1 - self.alpha
            ) * old_score + self.alpha * score
