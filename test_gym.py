import gymnasium as gym
env = gym.make('MountainCar-v0', render_mode="human")

env.reset()

done = False
while not done:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    done = terminated or truncated
    
env.close()