from environment.environment import Environment
from agents.base_agent import Base_Agent

def train_agent(agent: Base_Agent, env: Environment, episodes: int = 1, verbose: bool = False):
    all_rewards = []

    for ep in range(episodes):
        state = env.reset()
        agent.reset()
        done = False
        total_reward = 0

        while not done:
            action, action_id = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            if hasattr(agent, "remember"):
                agent.remember(state, action_id, reward)
            else:
                agent.update(action_id, reward)
            
            state = next_state
            total_reward += reward

        if hasattr(agent, "update") and hasattr(agent, "remember"):
            agent.update()

        all_rewards.append(total_reward)
        if verbose:
            print(f"🎯 Épisode {ep + 1}/{episodes} — Total reward: {total_reward:.2f}")

    return all_rewards