import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from rl_girp import GIRPEnv, compute_reward, make_features, setup_seed

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

# Update config
UPDATE_TIMESTEP = 500
GAMMA = 0.99
GAE_LAMBDA = 0.95
K_EPOCHS = 4
EPS_CLIP = 0.2
ENTROPY_COEF = 0.05

# Curriculum learning config
INITIAL_GOAL_HEIGHT = 1.5
GOAL_HEIGHT_INCREMENT = 1.0

WIN_RESULT_BUFFER_SIZE = 20
WIN_RATE_THRESHOLD = 0.7

# Training config
NUM_EPISODE = 10000

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

    def update(self, next_state=None):
        next_value = 0
        if next_state is not None:
            next_state = torch.FloatTensor(next_state).to(device)
            with torch.no_grad():
                next_value = self.policy.critic(next_state).item()

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(device)

        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        old_masks = torch.stack(self.buffer.masks, dim=0).detach().to(device)

        with torch.no_grad():
            values = self.policy.critic(old_states).squeeze()

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_non_terminal = 1.0 - is_terminals[t]
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - is_terminals[t]

            delta = rewards[t] + GAMMA * next_val * next_non_terminal - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_masks)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - ENTROPY_COEF * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return loss.mean().item()

    def save(self, checkpoint_path, score):
        torch.save({"model_state_dict": self.policy.state_dict(), "score": score}, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy_old.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("score", 0.0)


def run_train(writer: SummaryWriter) -> None:
    # Initialize the environment and set the goal
    env = GIRPEnv()
    env.open()
    goal_height = INITIAL_GOAL_HEIGHT
    env.set_goal_height(goal_height)

    # Determine the input dimension and initialize the PPO agent
    env_s, p_s = env.return_states()

    temp_obs = make_features(env_s, p_s)
    obs_dim = len(temp_obs)
    print(f"Observation Dimension: {obs_dim}")

    ppo_agent = PPO(obs_dim, ACT_DIM)

    # Load the pretrained model if a file exists
    if os.path.exists(LATEST_MODEL_PATH):
        best_eval_score = ppo_agent.load(LATEST_MODEL_PATH)
        print(f"Loaded PPO Model. Best Score: {best_eval_score}")
    else:
        best_eval_score = 0.0

    # Initialize variables for training
    best_global_score = best_eval_score
    best_episode_reward = -1e9
    time_step = 0

    # Start the main training loop
    win_result_buffer = [0.0 for _ in range(WIN_RESULT_BUFFER_SIZE)]
    for episode in range(1, NUM_EPISODE):
        # Periodically restart the environment to prevent lag
        if episode % 100 == 0:
            env.close()
            env.open()

        # Initialize episode variables and wait for the game to start
        best_episode_score = 0.0
        ep_reward = 0
        done = False
        actions = []
        episode_length = 0

        env.wait_for_game_reset()

        # Retrieve initial states and make features
        env_state, player_state = env.return_states()
        obs = make_features(env_state, player_state)

        # Run episode
        while not done:
            # Get action mask and select an action based on the policy
            mask = env.get_action_mask().to(device)
            action = ppo_agent.select_action(obs, mask)

            # Execute the selected action
            env.step(action)
            time_step += 1
            episode_length += 1

            # Observe the next state and make features
            temp_env, temp_p = env.return_states()
            next_obs = make_features(temp_env, temp_p)

            # Check termination conditions
            win_result_buffer[episode % WIN_RESULT_BUFFER_SIZE] = max(temp_env.get("winResult", 0), 0)
            if temp_env.get("winResult", 0) != 0:
                done = True

            if temp_env["time"] == 0 and env_state["time"] != 0:
                done = True

            # Save the model if the agent achieves a best score
            curr_score = temp_env["hiScore"]
            if curr_score > best_global_score:
                best_global_score = curr_score
                print(f"!!! Breakthrough! {curr_score:.2f} !!!")
                ppo_agent.save(MODEL_PATH, best_global_score)

            if curr_score > best_episode_score:
                best_episode_score = curr_score

            # Calculate step reward and update state variables
            step_reward = compute_reward(env_state, temp_env, action)

            player_state = temp_p
            env_state = temp_env

            # Store transition data in the buffer and update local variables
            actions.append(env.convert_action_to_key(action))
            ppo_agent.buffer.rewards.append(step_reward)
            ppo_agent.buffer.is_terminals.append(done)

            ep_reward += step_reward
            obs = next_obs

            # Perform PPO update if the timestep threshold is reached
            print(
                f"Update Countdown: {UPDATE_TIMESTEP - (time_step % UPDATE_TIMESTEP)} / "
                f"Action: {env.convert_action_to_key(action)}"
            )
            if time_step % UPDATE_TIMESTEP == 0:
                next_val_state = None if done else obs
                ppo_agent.update(next_state=next_val_state)

        win_rate = sum(win_result_buffer) / WIN_RESULT_BUFFER_SIZE
        if win_rate > WIN_RATE_THRESHOLD:
            ppo_agent.save(f"difficulty_increased_{goal_height}.pth", best_episode_reward)
            print(f"*** Difficulty Increased! Win Rate: {win_rate:.2f}, Current Goal Height: {goal_height:.2f} ***")

            win_result_buffer = [0.0 for _ in range(WIN_RESULT_BUFFER_SIZE)]

            goal_height += GOAL_HEIGHT_INCREMENT
            env.set_goal_height(goal_height)
        else:
            print(f"*** Current Win Rate: {win_rate:.2f} ***")

        # Log episode summaries and save the latest model
        writer.add_scalar("Episode/Reward", ep_reward, episode)
        writer.add_scalar("Episode/Score", best_episode_score, episode)
        writer.add_scalar("Episode/Win Rate", win_rate, episode)
        ppo_agent.save(LATEST_MODEL_PATH, best_episode_reward)

        # Update the best reward record and save the corresponding model
        if ep_reward > best_episode_reward:
            best_episode_reward = ep_reward
            ppo_agent.save(REWARD_MODEL_PATH, best_episode_reward)
            print(f"[✓] New Best Reward: {ep_reward:.2f}")


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("./log", current_time)
    writer = SummaryWriter(log_dir)

    setup_seed(SEED)

    run_train(writer)
