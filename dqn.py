# -*- coding: utf-8 -*-
import random
import gym, gym_anytrading
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    window_size = 10
    sample = 100
    env: gym.Env = gym.make('stocks-v0', frame_bound=(window_size, sample), window_size=window_size)
    state_size = env.observation_space.shape[0]  # 15
    # print(f'env.observation_space: {env.observation_space}')  # Box(-inf, inf, (15, 2), float32)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    # for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    action = env.action_space.sample()
    print(f'initial action: {action}')
    while True:
        # env.render()
        print('================')
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        action = agent.act(state)
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        print('--rnd_bot--')
        print(f'e: {agent.epsilon:.2}')
        print(f'position: {env._current_position}')
        print(f'reward: {reward}')
        print(f'total_reward: {env._total_reward}')
        print(f'reaction: {action}')
        if done:
            # print("episode: {}/{}, score: {}, e: {:.2}"
            #       .format(e, EPISODES, time, agent.epsilon))
            print("info:", info)
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

plt.cla()
env.render_all()
plt.show()
