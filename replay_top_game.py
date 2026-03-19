import json
import time
import numpy as np
from render_2048 import Game2048Renderer

def replay():
    try:
        with open("top_games.json", "r") as f:
            top_games = json.load(f)
    except:
        print("No top_games.json found! Run evaluate.py first.")
        return

    renderer = Game2048Renderer(size=4)

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
    replay()