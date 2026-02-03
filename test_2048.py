import gymnasium as gym
import gymnasium_2048
from render_2048 import Game2048Renderer
import time

env = gym.make("gymnasium_2048/TwentyFortyEight-v0", size=4)
renderer = Game2048Renderer(size=4)

obs, info = env.reset()

terminated = False
truncated = False


while not (terminated or truncated):
    renderer.render(obs,info["total_score"])
    action = env.action_space.sample() # joga aleatoriamente ate perder
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.3)


print(info["total_score"])
env.close()
