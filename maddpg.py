from mlagents.envs import UnityEnvironment
import numpy as np
import random
import datetime
from collections import deque
from agent import *
from network import *
from replay_buffer import *

game = 'Env'
env_name = "../"+ game + "/Elevator"

start_train_episode = 100
run_episode = 500
test_episode = 100

print_interval = 5
save_interval = 100

train_mode = True

n_agent = 4
episode_rewards = [0.0]
terminal_reward = []

if __name__ == '__main__':

    env = UnityEnvironment(file_name=env_name)
    default_brain = env.brain_names[0]
    success_cnt = 0
    step = 0
    rewards = deque(maxlen=print_interval)

    actor = ActorNetwork(input_dim=46, out_dim=3)
    critic = CriticNetwork(input_dim=46 + 5, out_dim=1)
    memory = MemoryBuffer(size=1000000)

    agent = Trainer(actor, critic, memory)

    # 각 에피소드를 거치며 replay memory에 저장
    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        env_info = env.reset(train_mode=train_mode)[default_brain]
        states = env_info.vector_observations

        done = False

        while not done:
            step += 1
            obs = agent.process_obs(states)
            actions = agent.get_exploration_action(obs)
            actions = agent.process_action(actions)
            env_info = env.step(actions)[default_brain]  # todo : local reward?
            next_states = env_info.vector_observations
            reward = env_info.rewards[0]  # global reward
            #print('actions : {}, reward :{}'.format(actions, reward))

            dones = env_info.local_done  # todo : done?

            episode_rewards[-1] += reward

            if train_mode:
                agent.memory.add(states, actions, reward, agent.process_obs(next_states), dones)

            states = next_states

            # train_mode 이고 일정 이상 에피소드가 지나면 학습
            if episode > start_train_episode and train_mode:
                agent.train_model()
            if all(dones):
                episode_rewards.append(0)
                terminal_reward.append(np.mean(rewards))

        rewards.append(episode_rewards)

        # 일정 이상의 episode를 진행 시 log 출력
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.3f} / success_cnt: {}".format
                  (step, episode, np.mean(rewards), success_cnt))
            success_cnt = 0

        # 일정 이상의 episode를 진행 시 현재 모델 저장
        if train_mode and episode % save_interval == 0 and episode != 0:
            print("model saved")
            agent.save_model()

    env.close()