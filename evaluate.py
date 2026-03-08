import gymnasium as gym
import gymnasium_2048
import time
import argparse
import numpy as np
from agent import DQNAgent
from render_2048 import Game2048Renderer

def evaluate(episodes, model_path, render=True):
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)
    agent = DQNAgent(input_dim=256, action_dim=env.action_space.n)
    
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    renderer = Game2048Renderer(size=4) if render else None

    scores = []
    max_tiles = []

    for ep in range(episodes):
        obs, info = env.reset()
        state = agent.preprocess_state(obs)
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if render:
                renderer.render(obs, info.get("total_score", 0))
                time.sleep(0.1)
                
            action = agent.select_action(state, env, eval_mode=True)
            obs_next, reward, terminated, truncated, info = env.step(action)
            
            # If the state didn't change (invalid action chosen deterministically), 
            # take a random action to prevent looping forever
            if np.array_equal(obs, obs_next) and not (terminated or truncated):
                action = env.action_space.sample()
                obs_next, reward, terminated, truncated, info = env.step(action)
                
            obs = obs_next
            state = agent.preprocess_state(obs)
            
        score = info.get("total_score", 0)
        
        powers = np.argmax(obs, axis=2)
        highest_power = np.max(powers)
        max_tile = int(2 ** highest_power) if highest_power > 0 else 0
        
        scores.append(score)
        max_tiles.append(max_tile)
        
        print(f"Eval Episode {ep+1}: Score={score}, Max Tile={max_tile}")
        
    env.close()
    
    print("\n--- Evaluation Results ---")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Tile Reached: {np.max(max_tiles)}")
    unique, counts = np.unique(max_tiles, return_counts=True)
    print("Tile Distribution:")
    for tile, count in zip(unique, counts):
        print(f"  {tile}: {count}/{episodes} ({(count/episodes)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model", type=str, default="dqn_2048_latest.pth")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    
    evaluate(args.episodes, args.model, render=not args.no_render)
