# TradeRL: A Comparative Study of Reinforcement Learning Algorithms for Automated Stock Trading

## Abstract
This project implements and compares various reinforcement learning algorithms for automated stock trading in a simulated environment. We explore Multi-Armed Bandit (MAB), Monte Carlo (MC), SARSA, Q-Learning, Deep Q-Network (DQN), and Proximal Policy Optimization (PPO) approaches to develop trading strategies for major tech stocks including TESLA, NVIDIA, and GAFAM companies (Google, Apple, Facebook/Meta, Amazon, Microsoft). Our findings demonstrate the relative strengths and limitations of each algorithm in the complex and dynamic stock market environment.

## 1. Introduction
Algorithmic trading has become increasingly prevalent in financial markets, with machine learning approaches gaining significant traction due to their ability to identify complex patterns and make data-driven decisions. Reinforcement learning (RL) presents a compelling framework for trading as it enables agents to learn optimal policies through trial-and-error interactions with the market environment. This project implements and evaluates several RL algorithms of increasing complexity to determine their efficacy in stock trading scenarios.

## 2. Environment and Data
### 2.1 Market Simulation
We developed a custom OpenAI Gym-compatible environment that simulates a trading platform. The environment tracks:
- Historical stock prices for six major tech companies (AAPL, AMZN, GOOGL, MSFT, NVDA, TSLA)
- Agent's cash balance
- Current stock holdings
- Total portfolio value

### 2.2 State Representation
The state consists of:
- Price history within a configurable window
- Current cash balance
- Current share holdings
- Overall portfolio value

### 2.3 Action Space
The agent can perform three types of actions for each stock:
- Buy: Allocate a portion of available cash to purchase shares
- Sell: Liquidate a portion of current holdings
- Hold: Maintain current position

### 2.4 Reward Signal
The reward is defined as the change in portfolio value between steps, encouraging the agent to maximize total returns.

## 3. Implemented Algorithms
### 3.1 Multi-Armed Bandit (MAB)
The simplest approach, treating each trading action as an arm of the bandit with no state consideration, using epsilon-greedy exploration.

### 3.2 Monte Carlo (MC) Method
Implements episodic learning with complete trajectories, using state abstraction to handle the complex market environment.

### 3.3 SARSA
A temporal difference learning approach that updates Q-values based on the action actually taken in the next state, balancing immediate rewards with expected future returns.

### 3.4 Q-Learning
An off-policy TD learning algorithm that learns the optimal policy regardless of the agent's actual actions, using epsilon-greedy exploration.

### 3.5 Deep Q-Network (DQN)
Extends Q-learning with neural networks to approximate the Q-function, incorporating experience replay and target networks to stabilize training.

### 3.6 Proximal Policy Optimization (PPO)
A policy gradient method that directly optimizes the policy while constraining updates to prevent destructively large policy changes.

## 4. Experimental Setup
### 4.1 Training Parameters
- Initial balance: $10,000
- Window size: 5 days of historical prices
- Training episodes: 100
- Epsilon values: 0.1 (greedy), 0.5 (optimal), 1.0 (random)
- Discount factor (gamma): 0.7-0.9

### 4.2 Evaluation Metrics
- Total accumulated reward
- Final portfolio value
- Risk-adjusted returns (Sharpe ratio)
- Algorithm convergence rates

## 5. Results and Discussion
### 5.1 Performance Comparison
Our experiments revealed several key insights:
- MAB agents achieved modest returns but showed high variance due to their stateless nature
- MC agents converged slowly but demonstrated robust performance in trending markets
- SARSA and Q-Learning exhibited faster convergence and better adaptation to market changes
- DQN showed superior performance in complex market conditions but required more computational resources
- PPO achieved the highest risk-adjusted returns by better managing the exploration-exploitation trade-off

### 5.2 Risk Analysis
The different algorithms exhibited varying risk profiles:
- MAB and MC agents showed higher volatility in returns
- TD-learning approaches (SARSA, Q-Learning) better balanced risk and reward
- DQN and PPO demonstrated more consistent performance across different market conditions

### 5.3 Limitations
- Market simulation simplifies many real-world trading constraints
- Transaction costs and slippage were not fully modeled
- Limited historical data may not capture all market conditions

## 6. Conclusion
This project demonstrates the application of reinforcement learning algorithms to stock trading, with more sophisticated approaches generally outperforming simpler methods. Our findings suggest that while modern deep RL methods like DQN and PPO offer superior performance, they come with increased complexity and computational requirements. The choice of algorithm should therefore be guided by the specific requirements of the trading strategy, available computational resources, and risk tolerance.

## 7. Future Work
- Incorporate market sentiment analysis from news and social media
- Explore multi-agent systems for competitive and collaborative trading
- Implement hierarchical RL approaches for long and short-term trading decisions
- Develop more sophisticated risk management strategies
- Test in live market conditions with paper trading

## References
1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
3. Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
4. Fischer, T. G. (2018). Reinforcement learning in financial markets - a survey. FAU Discussion Papers in Economics.
5. Zhang, Z., et al. (2020). Deep reinforcement learning for trading. The Journal of Financial Data Science, 2(2), 25-40.
