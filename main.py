from environment.environment import Environment
from agents.mab_agent import MAB_Agent
from train.train import train_agent
import csv
from argparse import ArgumentParser

def main(args):
    tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "TSLA"]

    data = {i: {t: float(row[t]) for t in tickers} \
        for i, row in enumerate(csv.DictReader( \
        open("data/nasdaq_stock_prices.csv", mode='r'), delimiter=','))
    }
    env = Environment(data, window_size=args.window_size, initial_balance=args.initial_balance, verbose=args)
    agent = MAB_Agent(env=env, epsilon=args.epsilon)
    train_agent(agent=agent, env=env, episodes=args.episodes, verbose=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--initial_balance", type=float, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    main(args)