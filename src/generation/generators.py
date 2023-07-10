import numpy as np
import numpy.typing as npt
import torch
from gymnasium.spaces import GraphInstance
from torch import nn
from torch.distributions import Categorical
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import trange

from .envs import G2SATEnv
from .graph import SATGraph


class G2SATPolicy:
    def __init__(
        self, env: G2SATEnv, model: nn.Module, random_state: int | None = None
    ) -> None:
        self.np_random = np.random.default_rng(random_state)
        self.env = env
        self.model = model

    @property
    def device(self):
        for param in self.model.parameters():
            return param.device

    def generate(
        self,
        template: npt.NDArray[np.int64] | None = None,
    ) -> SATGraph:
        if template is None:
            template = SATGraph.sample_template(
                self.env.num_vars, 3 * self.env.num_clauses, self.np_random
            )

        g = SATGraph.from_template(template)
        while not g.is_3sat():
            pairs = g.sample_valid_merges(self.env.num_sampled_pairs)
            if not pairs:
                break
            logits = self.predict_logits(
                g.to_graph_instance(),
                torch.as_tensor(np.array(pairs), device=self.device),
            )
            i, j = pairs[torch.argmax(logits)]
            g.merge(i, j)

        return g

    def predict_logits(
        self, instance: GraphInstance, pairs: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.model(
            torch.as_tensor(instance.nodes, device=self.device),
            edge_index=torch.as_tensor(instance.edge_links, device=self.device),
        )
        logits = torch.sum(embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], dim=-1)
        return logits


def train_reinforce(
    policy: G2SATPolicy,
    optimizer: Optimizer,
    num_episodes: int = 50_000,
    gamma: float = 0.99,
    sample_action: bool = False,
    writer: SummaryWriter | None = None,
) -> list[dict]:
    env = policy.env
    history = []
    pbar = trange(num_episodes, desc="Training", unit="episodes")
    for episode in pbar:
        log_probs = []
        rewards = []
        infos = []

        obs, info = env.reset()
        done = False
        while not done:
            action_logits = policy.predict_logits(
                obs["graph"],
                torch.as_tensor(obs["valid_actions"], device=policy.device),
            )
            if sample_action:
                dist = Categorical(logits=action_logits)
                action_idx = dist.sample()

                # Approximate the probability of choosing this pair
                # as the probability that the pair was in the samples
                # and the pair was chosen from the samples
                sample_size = len(obs["valid_actions"])
                total_pairs = obs["total_valid_actions"]
                if sample_size == total_pairs:
                    log_prob_pair_proposed = 0  # log(1)
                else:
                    log_prob_pair_proposed = np.log(sample_size) - np.log(total_pairs)
                log_prob_pair_chosen = dist.log_prob(action_idx)
                log_prob = log_prob_pair_proposed + log_prob_pair_chosen
            else:
                action_idx = torch.argmax(action_logits)

                # This is the log probability of the original model, in which
                # pairs are proposed sequentially
                log_prob_pair_proposed = -np.log(obs["total_valid_actions"])
                log_prob_pair_chosen = nn.functional.logsigmoid(
                    action_logits[action_idx]
                )
                log_prob = log_prob_pair_proposed + log_prob_pair_chosen
            log_probs.append(log_prob)

            action = obs["valid_actions"][action_idx]

            obs, reward, done, terminated, info = env.step(np.asarray(action))
            infos.append(info)
            rewards.append(reward)

        losses = []
        returns = []
        ret = 0
        for r in rewards[::-1]:
            ret = r + gamma * ret
            returns.append(ret)
        returns = torch.tensor(returns[::-1], device=policy.device)
        for log_prob, ret in zip(log_probs, returns):
            losses.append(-log_prob.view(1) * ret)

        loss = torch.cat(losses).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(
            {
                "infos": infos,
                "rewards": rewards,
                "log_probs": [p.item() for p in log_probs],
                "loss": loss.item(),
            }
        )

        if writer:
            writer.add_scalar("episode_loss", loss, global_step=episode)
            writer.add_scalar("return/shaped", returns[0], global_step=episode)
            writer.add_scalar("return/original", rewards[-1], global_step=episode)
            for k, v in infos[-1].items():
                writer.add_scalar("episode_info/" + k, v, global_step=episode)
        pbar.set_postfix(
            loss=loss.item(), return_shaped=returns[0], return_original=rewards[-1]
        )

    return history
