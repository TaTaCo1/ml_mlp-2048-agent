import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(metrics_file="training_metrics.json", save_path="training_plots.png", title = "2048 DQN Training Metrics"):
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
    fig.suptitle(title, fontsize=16)

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


def plot_evaluate_distributions(dqn_path, mcts_path, random_path, save_path="evaluation_distributions.png"):

    # load all 3
    with open(dqn_path, "r") as f:
        dqn_data = json.load(f)
    with open(mcts_path, "r") as f:
        mcts_data = json.load(f)
    with open(random_path, "r") as f:
        random_data = json.load(f)

    def extract(data):
        dist = data["tile_distribution"]
        tiles = sorted([int(k) for k in dist.keys()])
        pcts  = [dist[str(t)]["pct"] for t in tiles]
        return tiles, pcts

    random_tiles, random_pcts = extract(random_data)
    dqn_tiles,    dqn_pcts    = extract(dqn_data)
    mcts_tiles,   mcts_pcts   = extract(mcts_data)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Max Tile Distribution Comparison", fontsize=16)

    def plot_bar(ax, tiles, pcts, label, color, data):
        all_tiles = [32, 64, 128, 256, 512, 1024, 2048]

        tile_to_pct = {t: 0.0 for t in all_tiles}
        for tile, pct in zip(tiles, pcts):
            if tile in tile_to_pct:
                tile_to_pct[tile] = pct

        x = np.arange(len(all_tiles))
        pct_values = [tile_to_pct[t] for t in all_tiles]

        bars = ax.bar(x, pct_values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in all_tiles])
        ax.set_title(f"{label}\nAvg Score: {data['avg_score']:.0f} | Max Tile: {max(tiles)}")
        ax.set_xlabel("Max Tile Reached")
        ax.set_ylabel("% of Episodes")
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        for bar, pct in zip(bars, pct_values):
            if pct > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{pct:.1f}%",
                        ha='center', va='bottom', fontsize=9)

        max_tile = max(tiles)
        for bar, tile in zip(bars, all_tiles):
            if tile == max_tile:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)


    # plot each agent
    plot_bar(axs[0], random_tiles, random_pcts, "Random Agent",        "#7BF78C", random_data)
    plot_bar(axs[1], dqn_tiles,    dqn_pcts,    "DQN only",            "#E6D263", dqn_data)
    plot_bar(axs[2], mcts_tiles,   mcts_pcts,   "DQN trained with MCTS", "#558CF3", mcts_data)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dqn", type=str, default="training_metrics_dqn.json")
    parser.add_argument("--metrics_mcts", type=str, default="training_metrics_mcts.json")
    parser.add_argument("--save_dqn", type=str, default="training_plots_dqn.png")
    parser.add_argument("--save_mcts", type=str, default="training_plots_mcts.png")
    args = parser.parse_args()
    plot_metrics(args.metrics_dqn, args.save_dqn, "2048 DQN Training Metrics")
    plot_metrics(args.metrics_mcts, args.save_mcts, "2048 DQN+MCTS Training Metrics")
    plot_evaluate_distributions("distribution_dqn.json","distribution_mcts.json","distribution_random.json")
