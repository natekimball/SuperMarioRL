# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# import gym
# import DQNAgent

# env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
# env = JoypadSpace(env, SIMPLE_MOVEMENT)

# done = True
# env.reset()
# for step in range(5000):
#     action = env.action_space.sample()
#     print(action)
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

#     if done:
#        env.reset()

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from DQNAgent import DQNAgent
import numpy as np

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
EPISODES = 1000
agent = DQNAgent(state_size, action_size)

done = False
batch_size = 32

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(5000):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.train(batch_size)

