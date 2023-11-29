from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('SuperMarioBros-v3', apply_api_compatibility=True)#, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state_shape = env.observation_space.shape
action_size = env.action_space.n
EPISODES = 500
agent = DQNAgent(state_shape, action_size)

done = False
batch_size = 32

scores = []
moving_avg_scores = []
for e in range(EPISODES):
    state, info = env.reset()
    # image = np.reshape(state[0], [1, state_shape[0], state_shape[1], state_shape[2]])
    for time in range(5000):
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        reward = reward if not done else -10
        # next_state = np.reshape(next_state, [1, [state_shape[0], state_shape[1], state_shape[2]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score = info['score']
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    scores.append(score)
    moving_avg_score = np.mean(scores[-100:])  # calculate moving average of last 100 scores
    moving_avg_scores.append(moving_avg_score)
agent.save('mario-v1')

plt.figure(figsize=(10, 5))
plt.plot(scores, label='Score')
plt.plot(moving_avg_scores, label='Moving average (last 100)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.savefig('mario_scores.png')
plt.show()