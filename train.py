import random
import gymnasium as gym
import gymnasium_2048
import numpy as np
import os
import json
import argparse
import copy
import time
import torch
from dqn_agent import DQNAgent
from mcts import DQNAgentWithMCTS  # FIX 1: missing MCTS import


def _monotonicity_bonus(board):
    """
    Reward boards where tiles decrease from one corner outward.
    512  256  128  64
    32   16   8    4
    2    0    0    0
    0    0    0    0
    """
    bonus = 0
    for row in board:
        for i in range(len(row) - 1):
            if row[i] >= row[i+1]:
                bonus += 1
            else:
                bonus -= 1
    for col in board.T:
        for i in range(len(col) - 1):
            if col[i] >= col[i+1]:
                bonus += 1
            else:
                bonus -= 1
    return bonus


def train(episodes=1000, max_steps=2000, save_freq=128, resume=False,
          use_mcts=False, mcts_sims=20, mcts_depth=30):  # FIX 2: added mcts params

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Save frequency: {save_freq}")
    print(f"Resume: {resume}")
    print(f"Use MCTS: {use_mcts}")
    if use_mcts:
        print(f"  - Simulations: {mcts_sims}")
        print(f"  - Max Depth: {mcts_depth}")
    print("="*70 + "\n")

    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device, action_dim=env.action_space.n, network='cnn') 

    # FIX 3: mcts_agent creation was completely missing
    if use_mcts:
        mcts_kwargs = {
            'num_simulations': mcts_sims,
            'max_depth': mcts_depth,
        }
        mcts_agent = DQNAgentWithMCTS(agent, env, mcts_kwargs=mcts_kwargs)
        print(f"✓ Initialized MCTS wrapper (sims={mcts_sims}, depth={mcts_depth})")
    else:
        mcts_agent = None
        print(f"✓ Using standard epsilon-greedy DQN")

    metrics = {
        'rewards': [],
        'max_tiles': [],
        'losses': [],
        'scores': []
    }

    if resume:
        try:
            agent.load("dqn_2048_latest.pth")
            print(f"✓ Loaded previous model weights")
            if agent.epsilon < 0.02:
                agent.epsilon = 0.02
                print(f"✓ Reset epsilon to 0.02")
        except Exception as e:
            print(f"✗ Could not load previous model: {e}")

        try:
            if os.path.exists("training_metrics.json"):
                with open("training_metrics.json", "r") as f:
                    metrics = json.load(f)
                print(f"Loaded previous metrics with {len(metrics['scores'])} episodes.")
        except Exception as e:
            print(f"✗ Could not load previous metrics: {e}")

    start_episode = len(metrics['scores']) + 1
    total_episodes = start_episode + episodes - 1

    print(f"\nStarting training on {agent.device}")
    print(f"Episodes: {start_episode} → {total_episodes}\n")

    print(f"{'Episode':<10} {'Score':<12} {'Max Tile':<12} {'Reward':<12} {'Loss':<12} {'Epsilon':<10} {'Steps':<8}")
    print("-" * 80)
    best_avg_tile = 0.0
    for episode in range(start_episode, total_episodes + 1):
        obs, info = env.reset()
        state = agent.preprocess_state(obs)
        total_reward = 0
        total_loss = 0
        steps = 0
        next_obs = obs
        terminated = False
        truncated = False

        for step in range(max_steps):
            valid_actions = agent.get_valid_actions(env, state)

            if not valid_actions:
                terminated = True
                break

            # FIX 4: use mcts_agent when use_mcts is True
            if use_mcts:
                mcts_agent.env = env
                action = mcts_agent.select_action(state, use_mcts=True)
                if action is None:
                    terminated = True
                    break
            else:
                action = agent.select_action(state, env, eval_mode=False)
                if action is None:
                    terminated = True
                    break

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_state = agent.preprocess_state(next_obs)
            done = terminated or truncated

            board_now = np.argmax(next_state.reshape(4,4,16), axis=2)
            board_prev = np.argmax(state.reshape(4,4,16), axis=2)

            reward = reward * 0.3 + 2

            current_max = np.max(board_now)
            prev_max = np.max(board_prev)
            if current_max > prev_max:
                reward += 50 * (2 ** (current_max - prev_max))

            max_positions = list(zip(*np.where(board_now == current_max)))
            corners = [(0,0), (0,3), (3,0), (3,3)]
            if any(pos in corners for pos in max_positions):
                reward += 30

            reward += _monotonicity_bonus(board_now) * 2

            empty_cells = np.sum(board_now == 0)
            reward += empty_cells * 0.5

            if empty_cells <= 2:
                reward -= 5

            loss = agent.step(state, action, reward, next_state, done)
            total_loss += loss
            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        powers = np.argmax(next_obs, axis=2)
        highest_power = np.max(powers)
        max_tile = 2 ** highest_power if highest_power > 0 else 0

        avg_loss = total_loss / steps if steps > 0 else 0
        score = info.get("total_score", 0)

        metrics['rewards'].append(float(total_reward))
        metrics['max_tiles'].append(int(max_tile))
        metrics['losses'].append(float(avg_loss))
        metrics['scores'].append(float(score))
        metrics['epsilon'].append(float(agent.epsilon))

        print(f"{episode:<10} {score:<12.0f} {max_tile:<12} {total_reward:<12.2f} {avg_loss:<12.6f} {agent.epsilon:<10.4f} {steps:<8}")

        if episode % save_freq == 0:
            agent.save(f"dqn_2048_latest.pth")
            with open("training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n✓ Saved checkpoint at episode {episode}")

            recent_scores = metrics['scores'][-save_freq:]
            recent_tiles = metrics['max_tiles'][-save_freq:]
            print(f"  Last {save_freq} episodes:")
            print(f"    - Avg Score: {np.mean(recent_scores):.0f}")
            print(f"    - Avg Max Tile: {np.mean(recent_tiles):.0f}")
            print(f"    - Max Tile: {np.max(recent_tiles)}")
            print()

        if len(metrics['max_tiles']) >= 100:
            recent_avg = np.mean(metrics['max_tiles'][-100:])
            if not hasattr(agent, 'best_avg_tile'):
                agent.best_avg_tile = 0
            if recent_avg > agent.best_avg_tile:
                agent.best_avg_tile = recent_avg
                agent.save("dqn_2048_best.pth")
                print(f"  ★ New best avg tile: {recent_avg:.1f} → saved dqn_2048_best.pth")

    agent.save("dqn_2048_latest.pth")
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total episodes: {len(metrics['scores'])}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Average last 100 scores: {np.mean(metrics['scores'][-100:]):.0f}")
    print(f"Best score achieved: {np.max(metrics['scores']):.0f}")
    print(f"Highest tile reached: {np.max(metrics['max_tiles'])}")
    print("="*70 + "\n")

    env.close()
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--network", type=str, default='cnn',
                        choices=['cnn', 'mlp'])
    parser.add_argument("--use-mcts", action="store_true")  # FIX: added mcts args
    parser.add_argument("--mcts-sims", type=int, default=20)
    parser.add_argument("--mcts-depth", type=int, default=30)
    parser.add_argument("--save-freq", type=int, default=128)

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        resume=args.resume,
        use_mcts=args.use_mcts,
        mcts_sims=args.mcts_sims,
        mcts_depth=args.mcts_depth,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()