import gymnasium as gym
import gymnasium_2048
import time
import argparse
import numpy as np
from dqn_agent import DQNAgent
import torch
import json
from render_2048 import Game2048Renderer


# ─────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────
def _compute_max_tile(obs):
    powers = np.argmax(obs, axis=2)
    highest_power = int(np.max(powers))
    return 2 ** highest_power if highest_power > 0 else 0


def _save_top_games(all_episodes, top_n, filename):
    top_episodes = sorted(all_episodes, key=lambda x: x['score'], reverse=True)[:top_n]
    save_data = []
    for i, ep_data in enumerate(top_episodes):
        save_data.append({
            'rank': i + 1,
            'score': int(ep_data['score']),        # FIX: convert to int
            'max_tile': int(ep_data['max_tile']),   # FIX: convert to int
            'steps': int(ep_data['steps']),         # FIX: convert to int
            'frames': [
                (obs_frame.tolist(), int(score_frame), int(action) if action is not None else None)
                for obs_frame, score_frame, action in ep_data['frames']
            ]
        })
    with open(filename, "w") as f:
        json.dump(save_data, f)
    print(f"✓ Saved top {top_n} games to {filename}")


def _print_results(scores, max_tiles, episode_times, label):
    print("\n" + "="*70)
    print(f"{label} RESULTS")
    print("="*70)
    print(f"Average Score: {np.mean(scores):.2f} (±{np.std(scores):.2f})")
    print(f"Max Score:     {np.max(scores)}")
    print(f"Min Score:     {np.min(scores)}")
    print(f"\nMax Tile Distribution:")
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    for tile, count in sorted(zip(unique_tiles, counts), reverse=True):
        pct = (count / len(scores)) * 100
        bar = "█" * int(pct / 5)
        print(f"  {tile:6d}: {count:2d}/{len(scores)} ({pct:5.1f}%) {bar}")
    print(f"\nAverage Episode Time: {np.mean(episode_times):.2f}s")
    print("="*70 + "\n")
    return np.mean(scores), np.mean(max_tiles)

def _save_distribution(scores, max_tiles, episode_times, label, filename):
    """Save evaluation metrics and tile distribution to JSON."""
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    
    tile_distribution = {}
    for tile, count in zip(unique_tiles, counts):
        pct = (count / len(scores)) * 100
        tile_distribution[str(int(tile))] = {
            'count': int(count),
            'pct': round(float(pct), 2)
        }

    data = {
        'agent': label,
        'episodes': len(scores),
        'avg_score': round(float(np.mean(scores)), 2),
        'std_score': round(float(np.std(scores)), 2),
        'max_score': int(np.max(scores)),
        'min_score': int(np.min(scores)),
        'avg_max_tile': round(float(np.mean(max_tiles)), 2),
        'avg_episode_time': round(float(np.mean(episode_times)), 4),
        'tile_distribution': tile_distribution
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved distribution to {filename}")
# ─────────────────────────────────────────
# 1 — DQN only
# ─────────────────────────────────────────
def evaluate_dqn(episodes, model_path, top_n=3):
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
        max_tile = _compute_max_tile(obs)
        max_tiles.append(max_tile)
        all_episodes.append({'frames': episode_frames, 'score': score,
                              'max_tile': max_tile, 'steps': step_count})

        print(f"{ep+1:<10} {score:<12} {max_tile:<12} {episode_time:<10.2f} {step_count:<8}")

    env.close()
    avg_score, avg_tile = _print_results(scores, max_tiles, episode_times, "DQN")
    _save_top_games(all_episodes, top_n, "top_games_dqn.json")
    _save_distribution(scores, max_tiles, episode_times, "DQN", "distribution_dqn.json")
    return avg_score, avg_tile


# ─────────────────────────────────────────
# 2 — DQN + MCTS
# ─────────────────────────────────────────
def evaluate_dqn_mcts(episodes, model_path, top_n=3):
    print(f"\n{'='*70}")
    print(f"EVALUATING: DQN trained with MCTS")
    print(f"{'='*70}\n")

    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)
    agent = DQNAgent(device=None, action_dim=env.action_space.n, network='cnn')

    try:
        agent.load(model_path)
        print(f"✓ Loaded model from {model_path}\n")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        return 0, 0

    scores, max_tiles, episode_times, all_episodes = [], [], [], []

    print(f"{'Episode':<10} {'Score':<12} {'Max Tile':<12} {'Time':<10} {'Steps':<8}")
    print("-" * 60)

    for ep in range(episodes):
        episode_start = time.time()
        obs, info = env.reset()
        state = agent.preprocess_state(obs)
        terminated = False
        truncated = False
        step_count = 0
        episode_frames = [(obs.copy(), info.get("total_score", 0), None)]

        while not (terminated or truncated):
            # just DQN, no MCTS
            action = agent.select_action(state, env, eval_mode=True)
            if action is None:
                break
            obs_next, reward, terminated, truncated, info = env.step(action)
            obs = obs_next
            state = agent.preprocess_state(obs)
            step_count += 1
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
    avg_score, avg_tile = _print_results(scores, max_tiles, episode_times, "DQN trained with MCTS")
    _save_top_games(all_episodes, top_n, "top_games_mcts.json")
    _save_distribution(scores, max_tiles, episode_times, "DQN+MCTS", "distribution_mcts.json")
    return avg_score, avg_tile


# ─────────────────────────────────────────
# 3 — Random agent
# ─────────────────────────────────────────
def evaluate_random(episodes, top_n=3):
    print(f"\n{'='*70}")
    print(f"EVALUATING: Random agent")
    print(f"{'='*70}\n")

    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)

    scores, max_tiles, episode_times, all_episodes = [], [], [], []

    print(f"{'Episode':<10} {'Score':<12} {'Max Tile':<12} {'Time':<10} {'Steps':<8}")
    print("-" * 60)

    for ep in range(episodes):
        episode_start = time.time()
        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        episode_frames = [(obs.copy(), info.get("total_score", 0), None)]

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            episode_frames.append((obs.copy(), info.get("total_score", 0), action))

        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        score = info.get("total_score", 0)
        scores.append(score)
        max_tile = _compute_max_tile(obs)
        max_tiles.append(max_tile)
        all_episodes.append({'frames': episode_frames, 'score': score,
                              'max_tile': max_tile, 'steps': step_count})

        print(f"{ep+1:<10} {score:<12} {max_tile:<12} {episode_time:<10.2f} {step_count:<8}")

    env.close()
    avg_score, avg_tile = _print_results(scores, max_tiles, episode_times, "RANDOM")
    _save_top_games(all_episodes, top_n, "top_games_random.json")
    _save_distribution(scores, max_tiles, episode_times, "Random", "distribution_random.json")
    return avg_score, avg_tile


# ─────────────────────────────────────────
# Compare all 3
# ─────────────────────────────────────────
def compare_all(episodes, model_path, model_mcts_path, top_n=1):
    print("\n[1/3] DQN only...")
    dqn_score, dqn_tile = evaluate_dqn(episodes, model_path, top_n)

    print("\n[2/3] DQN + MCTS...")
    mcts_score, mcts_tile = evaluate_dqn_mcts(episodes, model_mcts_path,  # ← uses different model
                                               top_n)

    print("\n[3/3] Random agent...")
    rand_score, rand_tile = evaluate_random(episodes, top_n)

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
    parser.add_argument("--model", type=str, default="dqn_2048_best.pth",
                        help="Path to DQN only model")
    parser.add_argument("--model-mcts", type=str, default="dqn_2048_mcts_latest.pth",
                        help="Path to DQN+MCTS model")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top games to replay (default: 5)")
    parser.add_argument("--mcts-sims", type=int, default=30)
    parser.add_argument("--mcts-depth", type=int, default=15)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "dqn", "mcts", "random"],
                        help="Which agent to evaluate")

    args = parser.parse_args()

    if args.mode == "all":
        compare_all(args.episodes, args.model, args.model_mcts, args.top_n)
    elif args.mode == "dqn":
        evaluate_dqn(args.episodes, args.model, args.top_n)
    elif args.mode == "mcts":
        evaluate_dqn_mcts(args.episodes, args.model_mcts, args.top_n)
    elif args.mode == "random":
        evaluate_random(args.episodes, args.top_n)