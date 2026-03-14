"""
Training script with optional MCTS integration.
...
"""

import gymnasium as gym
import gymnasium_2048
import numpy as np
import os
import json
import argparse
import time
from agent import DQNAgent, DQNAgentWithMCTS


def train(episodes=1000, max_steps=2000, save_freq=100, resume=False, 
          use_mcts=False, mcts_sims=50, mcts_depth=20):

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Save frequency: {save_freq}")
    print(f"Resume: {resume}")
    print(f"Use MCTS: {use_mcts}")
    if use_mcts:
        print(f"  - Simulations per action: {mcts_sims}")
        print(f"  - Max tree depth: {mcts_depth}")
    print("="*70 + "\n")
    
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)
    agent = DQNAgent(input_dim=256, action_dim=env.action_space.n)
    
    if use_mcts:
        mcts_kwargs = {
            'num_simulations': mcts_sims,
            'max_depth': mcts_depth,
        }
        mcts_agent = DQNAgentWithMCTS(agent, env, mcts_kwargs=mcts_kwargs)
        print(f"✓ Initialized MCTS wrapper")
    else:
        mcts_agent = None
        print(f"✓ Using standard epsilon-greedy DQN")
    
    metrics = {
        'rewards': [],
        'max_tiles': [],
        'losses': [],
        'scores': [],
        'epsilon': []
    }
    
    if resume:
        try:
            agent.load("dqn_2048_latest.pth")
            print(f"✓ Loaded previous model weights")
        except Exception as e:
            print(f"✗ Could not load previous model: {e}")
        
        try:
            if os.path.exists("training_metrics.json"):
                with open("training_metrics.json", "r") as f:
                    metrics = json.load(f)
                print(f"✓ Loaded previous metrics ({len(metrics['scores'])} episodes)")
        except Exception as e:
            print(f"✗ Could not load previous metrics: {e}")
    
    start_episode = len(metrics['scores']) + 1
    total_episodes = start_episode + episodes - 1
    
    print(f"\nStarting training on {agent.device}")
    print(f"Episodes: {start_episode} → {total_episodes}\n")
    
    print(f"{'Episode':<10} {'Score':<12} {'Max Tile':<12} {'Reward':<12} {'Loss':<12} {'Epsilon':<10}")
    print("-" * 70)
    
    for episode in range(start_episode, total_episodes + 1):
        obs, info = env.reset()
        state = agent.preprocess_state(obs)
        
        total_reward = 0
        total_loss = 0
        steps = 0
        next_obs = obs  # FIX 1: initialize next_obs so max_tile never crashes
                        # if the episode ends on step 0

        for step in range(max_steps):
            if use_mcts:
                # FIX 2: update mcts_agent's env each episode so it always
                # simulates from the current environment state, not a stale copy
                mcts_agent.env = env
                action = mcts_agent.select_action(state, use_mcts=True)
            else:
                action = agent.select_action(state, env, eval_mode=False)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent.preprocess_state(next_obs)
            done = terminated or truncated
            
            # Reward shaping
            if np.array_equal(state, next_state):
                reward -= 10
            elif reward == 0:
                reward += 1
            
            loss = agent.step(state, action, reward, next_state, done)
            total_loss += loss
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # FIX 3: use next_obs directly — argmax over axis=2 is correct for
        # one-hot encoded boards, but next_obs could be 0 everywhere on the
        # very first step, so clamp highest_power to at least 0
        powers = np.argmax(next_obs, axis=2)
        highest_power = int(np.max(powers))
        max_tile = 2 ** highest_power if highest_power > 0 else 0
        
        avg_loss = total_loss / steps if steps > 0 else 0
        score = info.get("total_score", 0)
        
        metrics['rewards'].append(float(total_reward))
        metrics['max_tiles'].append(int(max_tile))
        metrics['losses'].append(float(avg_loss))
        metrics['scores'].append(float(score))
        metrics['epsilon'].append(float(agent.epsilon))
        
        print(f"{episode:<10} {score:<12.0f} {max_tile:<12} {total_reward:<12.2f} {avg_loss:<12.6f} {agent.epsilon:<10.4f}")
        
        if episode % save_freq == 0:
            agent.save("dqn_2048_latest.pth")
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


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent on 2048")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-mcts", action="store_true")
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--mcts-depth", type=int, default=15)
    parser.add_argument("--save-freq", type=int, default=100)
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        resume=args.resume,
        use_mcts=args.use_mcts,
        mcts_sims=args.mcts_sims,
        mcts_depth=args.mcts_depth,
        save_freq=args.save_freq
    )


if __name__ == "__main__":
    main()