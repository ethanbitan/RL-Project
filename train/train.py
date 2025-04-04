from environment.environment import Environment
from agents.base_agent import Base_Agent
from agents.mab_agent import MAB_Agent
from agents.mc_agent import MC_Agent
from agents.sarsa_agent import SARSA_Agent
from agents.dqn_agent import DQN_Agent

def train_and_eval_agent(agent: Base_Agent, env: Environment, episodes: int = 1, train: bool = True, verbose: bool = False):
    all_rewards = []
    env.train_mode = train

    for ep in range(episodes):
        state = env.reset()
        agent.reset()
        done = False
        total_reward = 0

        if isinstance(agent, MAB_Agent):
            while not done:
                action, action_id = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(action_id, reward)
                state = next_state
                total_reward += reward

        elif isinstance(agent, MC_Agent):
            while not done:
                action, action_id = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action_id, reward)
                state = next_state
                total_reward += reward
            agent.update()

        elif isinstance(agent, SARSA_Agent):
            action, action_id = agent.act(state)
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action, next_action_id = agent.act(next_state)
                agent.update(state, action_id, reward, next_state, next_action_id)
                state, action, action_id = next_state, next_action, next_action_id
                total_reward += reward

        elif isinstance(agent, DQN_Agent):
            while not done:
                action, action_id = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action_id, reward, next_state, done)
                agent.update()
                state = next_state
                total_reward += reward

        else:
            raise NotImplementedError("Agent type not supported.")

        all_rewards.append(total_reward)
        if verbose:
            phase = "Training" if train else "Testing"
            print(f"{phase} | Épisode {ep + 1}/{episodes} — Total reward: {total_reward:.2f}")

    return all_rewards