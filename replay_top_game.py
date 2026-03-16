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

    print(f"\nFound {len(top_games)} saved games\n")

    for game in top_games:
        print(f"Game {game['rank']} | Score: {game['score']} | Max Tile: {game['max_tile']} | Steps: {game['steps']}")
        print("Press Ctrl+C to skip\n")

        try:
            for frame_data in game['frames']:
                obs_list, score_frame, last_action = frame_data
                obs_frame = np.array(obs_list)
                renderer.render(obs_frame, score_frame, last_action=last_action)
                time.sleep(0.3)
        except KeyboardInterrupt:
            print("Skipped!\n")
            continue

        print(f"Game {game['rank']} finished!\n")
        time.sleep(1)

if __name__ == "__main__":
    replay()