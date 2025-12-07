import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_girp import GIRPEnv, make_features, setup_seed

# ==========================================
# [Hyperparameters for PPO]
# ==========================================
MODEL_PATH = "best_ppo_model_final.pth"
REWARD_MODEL_PATH = "best_reward_model_final.pth"
LATEST_MODEL_PATH = "latest_model.pth"

SEED = 42

# PPO config
ACT_DIM = 27
LR_ACTOR = 0.0001
LR_CRITIC = 0.001

# Training config
NUM_EVAL_EPISODES = 20

# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor Network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, action_dim)
        )

        # Critic Network (Value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)

        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0.0)

        self.action_logits = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask):
        action_logits = self.actor(state)

        if mask is not None:
            action_logits = action_logits + (mask - 1.0) * 1e9

        dist = Categorical(logits=action_logits)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        self.action_logits = action_logits

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, mask):
        action_logits = self.actor(state)

        if mask is not None:
            action_logits = action_logits + (mask - 1.0) * 1e9

        dist = Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": LR_ACTOR},
                {"params": self.policy.critic.parameters(), "lr": LR_CRITIC},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = RolloutBuffer()
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, mask):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state, mask)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.masks.append(mask)

        return action.item()

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy_old.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("score", 0.0)


def run_eval(eval_model_path):
    env = GIRPEnv()
    env.open()

    env_s, p_s = env.return_states()
    temp_obs = make_features(env_s, p_s)
    obs_dim = len(temp_obs)

    agent = PPO(obs_dim, ACT_DIM)
    if os.path.exists(LATEST_MODEL_PATH):
        best_eval_score = agent.load(eval_model_path)
        print(f"Loaded PPO Model. Best Score: {best_eval_score}")

    total_rewards = []
    success_count = 0

    for episode in range(1, NUM_EVAL_EPISODES + 1):
        env.wait_for_game_reset()

        env_state, player_state = env.return_states()
        obs = make_features(env_state, player_state)

        done = False
        step_count = 0

        print(f"--- Episode {episode} Start ---")

        while not done:
            mask = env.get_action_mask().to(device)

            action = agent.select_action(obs, mask)

            env.step(action)
            step_count += 1

            temp_env, temp_p = env.return_states()
            next_obs = make_features(temp_env, temp_p)

            if temp_env.get("winResult", 0) != 0:
                done = True
                if temp_env["winResult"] == 1:
                    print(f"Episode {episode} Cleared! (Win)")
                    success_count += 1

            if temp_env["time"] == 0 and env_state["time"] != 0:
                done = True

            env_state = temp_env
            player_state = temp_p
            obs = next_obs

        final_score = env_state["hiScore"]
        print(f"Episode {episode} Finished. Steps: {step_count}, Score: {final_score:.2f}")
        total_rewards.append(final_score)

    # 4. Final Summary
    avg_score = np.mean(total_rewards)

    print("\n" + "=" * 30)
    print(f"EVALUATION RESULT ({NUM_EVAL_EPISODES} Episodes)")
    print(f"Average Score: {avg_score:.2f}")
    print("=" * 30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    eval_model_path = args.path

    setup_seed(SEED)

    run_eval(eval_model_path)
