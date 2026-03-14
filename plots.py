import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(metrics_file="training_metrics.json", save_path="training_plots.png"):
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found. Train the agent first.")
        return

    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Could not load {metrics_file}: {e}")
        return

    episodes = len(metrics['scores'])
    if episodes == 0:
        print("No metrics to plot.")
        return

    x = np.arange(1, episodes + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('2048 DQN Training Metrics', fontsize=16)

    axs[0, 0].plot(x, metrics['scores'], alpha=0.3, color='blue', label='Score')
    window = min(episodes // 10, 50) if episodes > 10 else 0
    if window > 0 and episodes > window:
        ma_scores = np.convolve(metrics['scores'], np.ones(window)/window, mode='valid')
        axs[0, 0].plot(np.arange(window, episodes + 1), ma_scores, color='darkblue', label=f'{window}-ep MA')
    axs[0, 0].set_title('Score per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Score')
    axs[0, 0].legend()

    # 2. Rewards
    axs[0, 1].plot(x, metrics['rewards'], alpha=0.3, color='green', label='Reward')
    if window > 0 and episodes > window:
        ma_rewards = np.convolve(metrics['rewards'], np.ones(window)/window, mode='valid')
        axs[0, 1].plot(np.arange(window, episodes + 1), ma_rewards, color='darkgreen', label=f'{window}-ep MA')
    axs[0, 1].set_title('Total Reward per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Reward')
    axs[0, 1].legend()

    # 3. Max Tiles
    axs[1, 0].scatter(x, metrics['max_tiles'], alpha=0.3, color='orange', marker='.')
    axs[1, 0].set_yscale('log', base=2)
    # Handle zero tiles gracefully
    nonzero_tiles = [t for t in metrics['max_tiles'] if t > 0]
    if nonzero_tiles:
        tiles = sorted(list(set(nonzero_tiles)))
        axs[1, 0].set_yticks(tiles)
        axs[1, 0].set_yticklabels([str(t) for t in tiles])
    axs[1, 0].set_title('Max Tile per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Max Tile')
    
    # 4. Loss
    axs[1, 1].plot(x, metrics['losses'], alpha=0.3, color='red', label='Loss')
    if window > 0 and episodes > window:
        ma_losses = np.convolve(metrics['losses'], np.ones(window)/window, mode='valid')
        axs[1, 1].plot(np.arange(window, episodes + 1), ma_losses, color='darkred', label=f'{window}-ep MA')
    axs[1, 1].set_title('Average Loss per Episode')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="training_metrics.json")
    parser.add_argument("--save", type=str, default="training_plots.png")
    args = parser.parse_args()
    
    plot_metrics(args.metrics, args.save)
