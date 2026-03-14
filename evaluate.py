"""
Evaluation script for DQN agent with optional MCTS.

Usage:
    # Evaluate DQN only (baseline)
    python evaluate_with_mcts.py --model model.pth --episodes 5
    
    # Evaluate with MCTS
    python evaluate_with_mcts.py --model model.pth --episodes 5 --use-mcts --mcts-sims 100
    
    # Evaluate with reduced MCTS (faster)
    python evaluate_with_mcts.py --model model.pth --episodes 5 --use-mcts --mcts-sims 50
"""

import gymnasium as gym
import gymnasium_2048
import time
import argparse
import numpy as np
from agent import DQNAgent, DQNAgentWithMCTS
import torch
from render_2048 import Game2048Renderer


def evaluate(episodes, model_path, use_mcts=False, mcts_sims=100, mcts_depth=20, 
             mcts_ucb=None, render=True, verbose=False):
    """
    Evaluate agent on 2048 environment.
    
    Args:
        episodes: Number of evaluation episodes
        model_path: Path to saved DQN model
        use_mcts: If True, use MCTS for decision-making
        mcts_sims: Number of MCTS simulations
        mcts_depth: Maximum tree depth for MCTS
        mcts_ucb: UCB constant (None = sqrt(2) ≈ 1.41)
        render: If True, render the game
        verbose: Print detailed MCTS statistics
    """
    print(f"\n{'='*70}")
    print(f"Evaluation Setup")
    print(f"{'='*70}")
    print(f"Environment: 2048 (4x4)")
    print(f"Episodes: {episodes}")
    print(f"Model: {model_path}")
    print(f"Use MCTS: {use_mcts}")
    if use_mcts:
        print(f"  - Simulations: {mcts_sims}")
        print(f"  - Max Depth: {mcts_depth}")
        print(f"  - UCB Constant: {mcts_ucb or 'sqrt(2) ≈ 1.41'}")
    print(f"Render: {render}")
    print(f"{'='*70}\n")
    
    # Create environment
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)
    
    # Create DQN agent
    dqn_agent = DQNAgent(input_dim=256, action_dim=env.action_space.n)
    
    # Load model
    try:
        dqn_agent.load(model_path)
        print(f"✓ Loaded model from {model_path}\n")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        return
    
    if use_mcts:
        mcts_kwargs = {
            'num_simulations': mcts_sims,
            'max_depth': mcts_depth,
        }
        if mcts_ucb is not None:
            mcts_kwargs['ucb_constant'] = mcts_ucb
        
        agent = DQNAgentWithMCTS(dqn_agent, env, mcts_kwargs=mcts_kwargs)
        print(f"✓ Initialized MCTS with {mcts_sims} simulations\n")
    else:
        agent = dqn_agent
        print(f"✓ Using DQN baseline\n")
    
    # Setup renderer if available
    renderer = Game2048Renderer(size=4)
    
    # Metrics
    scores = []
    max_tiles = []
    episode_times = []
    
    # Evaluation loop
    print(f"{'Episode':<10} {'Score':<12} {'Max Tile':<12} {'Time':<10} {'Status':<15}")
    print("-" * 60)
    
    for ep in range(episodes):
        episode_start = time.time()
        
        obs, info = env.reset()
        state = dqn_agent.preprocess_state(obs)
        terminated = False
        truncated = False
        step_count = 0
        mcts_time = 0
        
        # Episode loop
        while not (terminated or truncated):
            # Render
            if renderer:
                renderer.render(obs, info.get("total_score", 0))
                time.sleep(0.05)
            
            # Select action
            if use_mcts:
                mcts_start = time.time()
                action = agent.select_action(state, use_mcts=True, verbose=False)
                mcts_time += time.time() - mcts_start
            else:
                action = agent.select_action(state, env, eval_mode=True)
            
            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action)
            
            # Handle invalid actions (state doesn't change)
            if np.array_equal(obs, obs_next) and not (terminated or truncated):
                action = env.action_space.sample()
                obs_next, reward, terminated, truncated, info = env.step(action)
            
            obs = obs_next
            state = dqn_agent.preprocess_state(obs)
            step_count += 1
        
        # Collect episode statistics
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        score = info.get("total_score", 0)
        scores.append(score)
        
        # Compute max tile
        powers = np.argmax(obs, axis=2)
        highest_power = np.max(powers)
        max_tile = int(2 ** highest_power) if highest_power > 0 else 0
        max_tiles.append(max_tile)
        
        # Print episode result
        avg_sim_time = (mcts_time / step_count * 1000) if use_mcts and step_count > 0 else 0
        time_str = f"{episode_time:.2f}s"
        if use_mcts and step_count > 0:
            time_str += f" ({avg_sim_time:.0f}ms/sim)"
        
        status = f"Steps: {step_count}"
        print(f"{ep+1:<10} {score:<12} {max_tile:<12} {time_str:<10} {status:<15}")
    
    env.close()
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
    
    print(f"\nAverage Episode Time: {np.mean(episode_times):.2f}s (±{np.std(episode_times):.2f}s)")
    
    if use_mcts:
        print("\n" + "="*70)
        print("MCTS STATISTICS")
        print("="*70)
        print(f"Simulations per episode: {mcts_sims}")
        print(f"Average time per episode: {np.mean(episode_times):.2f}s")
        print(f"Note: Time includes rendering. Without rendering: ~2-5x faster")
    
    print("="*70 + "\n")


def compare_baseline_vs_mcts(episodes_each, model_path, mcts_sims=100, render=False):
    """
    Compare baseline DQN vs DQN+MCTS performance.
    """
    print("\n" + "="*70)
    print("COMPARISON: DQN vs DQN+MCTS")
    print("="*70)
    
    # Evaluate baseline
    print("\n[1/2] Evaluating DQN baseline...")
    evaluate(episodes_each, model_path, use_mcts=False, render=render)
    
    # Evaluate with MCTS
    print("\n[2/2] Evaluating DQN+MCTS...")
    evaluate(episodes_each, model_path, use_mcts=True, mcts_sims=mcts_sims, render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DQN agent with optional MCTS"
    )
    parser.add_argument("--episodes", type=int, default=1, 
                        help="Number of evaluation episodes")
    parser.add_argument("--model", type=str, default="dqn_2048_latest.pth",
                        help="Path to saved DQN model")
    parser.add_argument("--use-mcts", action="store_true",
                        help="Use MCTS for decision-making")
    parser.add_argument("--mcts-sims", type=int, default=100,
                        help="Number of MCTS simulations (default: 100)")
    parser.add_argument("--mcts-depth", type=int, default=20,
                        help="Max tree depth for MCTS (default: 20)")
    parser.add_argument("--mcts-ucb", type=float, default=None,
                        help="UCB constant for MCTS (default: sqrt(2) ≈ 1.41)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed MCTS statistics")
    parser.add_argument("--compare", action="store_true",
                        help="Compare DQN vs DQN+MCTS")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_baseline_vs_mcts(args.episodes, args.model, args.mcts_sims, 
                                render=not args.no_render)
    else:
        evaluate(args.episodes, args.model, use_mcts=args.use_mcts, 
                mcts_sims=args.mcts_sims, mcts_depth=args.mcts_depth,
                mcts_ucb=args.mcts_ucb, render=not args.no_render, 
                verbose=args.verbose)