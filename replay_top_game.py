import json
import time
import numpy as np
import pygame
from render_2048 import Game2048TripleRenderer


def load_games(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ File not found: {filename}")
        return None


def replay_all_side_by_side(delay=0.15):
    """Replay all 3 agents side by side in one window."""
    dqn_games    = load_games("top_games_dqn.json")
    mcts_games   = load_games("top_games_mcts.json")
    random_games = load_games("top_games_random.json")

    if not dqn_games or not mcts_games or not random_games:
        print("Missing files! Run evaluate.py --mode all first.")
        return

    renderer = Game2048TripleRenderer(size=4, tile_size=100)

    num_games = min(len(dqn_games), len(mcts_games), len(random_games))

    for i in range(num_games):
        dqn_game    = dqn_games[i]
        mcts_game   = mcts_games[i]
        random_game = random_games[i]

        # FIX 1: duplicate print removed
        print(f"Game {i+1}/{num_games}")
        print(f"  Random:   Score={random_game['score']} Max Tile={random_game['max_tile']}")
        print(f"  DQN:      Score={dqn_game['score']}    Max Tile={dqn_game['max_tile']}")
        print(f"  MCTS:     Score={mcts_game['score']}   Max Tile={mcts_game['max_tile']}")
        print("Press Ctrl+C to skip\n")

        dqn_frames    = dqn_game['frames']
        mcts_frames   = mcts_game['frames']
        random_frames = random_game['frames']

        max_len = max(len(dqn_frames), len(mcts_frames), len(random_frames))

        try:
            for fi in range(max_len):
                dqn_f    = dqn_frames[min(fi, len(dqn_frames) - 1)]
                mcts_f   = mcts_frames[min(fi, len(mcts_frames) - 1)]
                random_f = random_frames[min(fi, len(random_frames) - 1)]

                obs_dqn,    score_dqn,    action_dqn    = dqn_f
                obs_mcts,   score_mcts,   action_mcts   = mcts_f
                obs_random, score_random, action_random = random_f

                renderer.render(
                    np.array(obs_random), score_random, action_random, "Random",
                    np.array(obs_dqn),    score_dqn,    action_dqn,    "DQN",
                    np.array(obs_mcts),   score_mcts,   action_mcts,   "DQN+MCTS"
                )
                time.sleep(delay)

        except KeyboardInterrupt:
            print("Skipped!\n")
            continue

        print(f"Game {i+1} finished!\n")
        time.sleep(1)  # FIX 2: pause between games was missing

    # FIX 3: keep window open after all games finish
    # wait until user closes the window manually
    print("All games finished! Close the window to exit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replay top games side by side")
    parser.add_argument("--delay", type=float, default=0.15,
                        help="Seconds between frames (default: 0.15)")
    args = parser.parse_args()

    replay_all_side_by_side(args.delay)