import gymnasium as gym
import gymnasium_2048
import time
import argparse
import numpy as np
from dqn_agent import DQNAgent
import torch
import json
from render_2048 import Game2048Renderer


def evaluate(episodes, model_path, render=False, verbose=False, top_n=3):  # default changed to 3

    print(f"\n{'='*70}")
    print(f"Evaluation Setup")
    print(f"{'='*70}")
    print(f"Environment: 2048 (4x4)")
    print(f"Episodes: {episodes}")
    print(f"Model: {model_path}")
    print(f"Top games to replay: {top_n}")
    print(f"{'='*70}\n")

    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)
    agent = DQNAgent(device=None, action_dim=env.action_space.n, network='cnn')

    try:
        agent.load(model_path)
        print(f"✓ Loaded model from {model_path}\n")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    scores = []
    max_tiles = []
    episode_times = []
    all_episodes = []

    print(f"{'Episode':<10} {'Score':<12} {'Max Tile':<12} {'Time':<10} {'Steps':<8}")
    print("-" * 60)

    # ── Phase 1: play episodes silently ──
    for ep in range(episodes):
        episode_start = time.time()

        obs, info = env.reset()
        state = agent.preprocess_state(obs)
        terminated = False
        truncated = False
        step_count = 0

        episode_frames = [(obs.copy(), info.get("total_score", 0), None)]  # first frame has no action

        while not (terminated or truncated):
            action = agent.select_action(state, env, eval_mode=True)

            if action is None:
                break

            obs_next, reward, terminated, truncated, info = env.step(action)
            obs = obs_next
            state = agent.preprocess_state(obs)
            step_count += 1

            # store action with frame
            episode_frames.append((obs.copy(), info.get("total_score", 0), action))


        episode_time = time.time() - episode_start
        episode_times.append(episode_time)

        score = info.get("total_score", 0)
        scores.append(score)

        powers = np.argmax(obs, axis=2)
        highest_power = int(np.max(powers))
        max_tile = 2 ** highest_power if highest_power > 0 else 0
        max_tiles.append(max_tile)

        all_episodes.append({
            'frames': episode_frames,
            'score': score,
            'max_tile': max_tile,
            'steps': step_count
        })

        print(f"{ep+1:<10} {score:<12} {max_tile:<12} {episode_time:<10.2f} {step_count:<8}")

    env.close()

    if not scores:
        print("No episodes completed.")
        return

    # ── Phase 2: print results ──
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Episodes: {episodes}")
    print(f"Average Score: {np.mean(scores):.2f} (±{np.std(scores):.2f})")
    print(f"Max Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")
    print(f"\nMax Tile Distribution:")

    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    for tile, count in sorted(zip(unique_tiles, counts), reverse=True):
        pct = (count / episodes) * 100
        bar = "█" * int(pct / 5)
        print(f"  {tile:6d}: {count:2d}/{episodes} ({pct:5.1f}%) {bar}")

    print(f"\nAverage Episode Time: {np.mean(episode_times):.2f}s")
    print("="*70 + "\n")

    # ── Phase 3: save top N games to JSON ──
    top_episodes = sorted(all_episodes, key=lambda x: x['score'], reverse=True)[:top_n]

    save_data = []
    for i, ep_data in enumerate(top_episodes):
        save_data.append({
            'rank': i + 1,
            'score': ep_data['score'],
            'max_tile': ep_data['max_tile'],
            'steps': ep_data['steps'],
            'frames': [
                (obs_frame.tolist(), score_frame, action)
                for obs_frame, score_frame, action in ep_data['frames']
            ]
        })

    with open("top_games.json", "w") as f:
        json.dump(save_data, f)

    print(f"✓ Saved top {top_n} games to top_games.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN agent on 2048")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model", type=str, default="dqn_2048_best.pth")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top games to replay (default: 5)")

    args = parser.parse_args()

    evaluate(
        args.episodes,
        args.model,
        verbose=args.verbose,
        top_n=args.top_n
    )