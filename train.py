import gymnasium as gym
import gymnasium_2048
import numpy as np
import os
import json
from agent import DQNAgent

def train(episodes=1000, max_steps=2000, save_freq=100, resume=False):
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)
    agent = DQNAgent(input_dim=256, action_dim=env.action_space.n)
    
    metrics = {
        'rewards': [],
        'max_tiles': [],
        'losses': [],
        'scores': []
    }
    
    if resume:
        try:
            agent.load("dqn_2048_latest.pth")
            print("Loaded previous model weights from dqn_2048_latest.pth")
        except Exception as e:
            print(f"Could not load previous model: {e}")
            
        try:
            if os.path.exists("training_metrics.json"):
                with open("training_metrics.json", "r") as f:
                    metrics = json.load(f)
                print(f"Loaded previous metrics with {len(metrics['scores'])} episodes.")
        except Exception as e:
            print(f"Could not load previous metrics: {e}")
    
    start_episode = len(metrics['scores']) + 1
    total_episodes = start_episode + episodes - 1
    
    print(f"Starting training on {agent.device} for {episodes} new episodes...")
    
    for episode in range(start_episode, total_episodes + 1):
        obs, info = env.reset()
        state = agent.preprocess_state(obs)
        
        total_reward = 0
        total_loss = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_state = agent.preprocess_state(next_obs)
            done = terminated or truncated
            
            # Simple reward shaping: give small penalty for invalid actions to encourage valid moves
            # But the env handles it by not changing the state. If state == next_state, penalty.
            if np.array_equal(state, next_state):
                reward -= 10
            elif reward == 0:
                reward += 1 # small reward for keeping the game alive
            
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
        
        print(f"Episode {episode}/{episodes} | Score: {score} | Max Tile: {max_tile} | Reward: {total_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")
        
        if episode % save_freq == 0:
            agent.save(f"dqn_2048_latest.pth")
            with open("training_metrics.json", "w") as f:
                json.dump(metrics, f)
            print(f"Saved model and metrics at episode {episode}")

    # Save at the end of training always
    agent.save(f"dqn_2048_latest.pth")
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"Saved final model and metrics at end of training!")

    env.close()
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest saved model and metrics")
    args = parser.parse_args()
    train(episodes=args.episodes, resume=args.resume)
