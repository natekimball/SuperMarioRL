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

state_shape = env.observation_space.shape
print(state_shape)
action_size = env.action_space.n
print(action_size)
EPISODES = 1000
agent = DQNAgent(state_shape, action_size)

done = False
batch_size = 32

for e in range(EPISODES):
    state = env.reset()[0]
    # image = np.reshape(state[0], [1, state_shape[0], state_shape[1], state_shape[2]])
    for time in range(5000):
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        reward = reward if not done else -10
        # next_state = np.reshape(next_state, [1, [state_shape[0], state_shape[1], state_shape[2]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

