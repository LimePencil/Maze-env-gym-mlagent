import sys

import torch
from ReplayBuffer import ReplayBuffer
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch.optim as optim
from PIL import Image
import time
import random
import torch.nn.functional as F
from DQN import DQN
from torch.utils.tensorboard import SummaryWriter


# TODO: memory allocation need to be fixed
class Agent:
    def __init__(self):
        # using GPU
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

        # initializing model
        self.train_mode = True
        self.load = False
        self.path_to_save_file = ""
        self.number_of_actions = 8
        self.q_net = DQN(self.number_of_actions)
        self.q_net.to(self.device)
        self.q_target_net = DQN(self.number_of_actions)
        self.q_target_net.to(self.device)
        self.learning_rate = 0.00025
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.loss = None
        # new environment from .exe file
        path_to_env = "envs/Ml-agent-with-gym"
        unity_env = UnityEnvironment(path_to_env, no_graphics=True)
        self.env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)

        self.writer = SummaryWriter('runs/maze_test_dqn_1')

        self.rewards = []
        self.epi_length = []

        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.epsilon_max_frame = 1000000

        buffer_size_limit = 10000  # need to be set so that it does not go over the memory limit
        self.number_of_episode = 10000
        self.target_update_step = 10000
        self.print_interval = 100
        self.save_interval = 500
        self.replay_start_size = 50000
        self.batch_size = 32
        self.gamma = 0.99
        self.total_step = 0
        self.memory = ReplayBuffer(self.batch_size, buffer_size_limit)
        # loading from save file TODO: not sure this works yet needs to be tested
        if self.load:
            checkpoint = torch.load(self.path_to_save_file)
            self.q_net = checkpoint['policy_net']
            self.q_target_net = checkpoint['target_net']
            self.memory.buffer = checkpoint['replay_memory']
            self.optimizer = checkpoint['optimizer']

    def main(self):
        for epi in range(self.number_of_episode):
            start_time = time.time()
            state = self.env.reset()
            state = self.arr_to_tensor(state)
            cumulative_reward = 0
            step = 0
            while True:
                action = self.change_action_to_continuous(self.get_action(state))
                next_state, reward, done, info = self.env.step(action)
                cumulative_reward += reward
                if self.train_mode:
                    self.memory.push((state, action, reward, next_state, done))
                else:
                    self.env.render()
                    time.sleep(0.01)

                if done:
                    self.rewards.append(cumulative_reward)
                    self.epi_length.append(step)
                    # writing to tensorboard
                    self.writer.add_scalar("Rewards", cumulative_reward, self.total_step)
                    self.writer.add_scalar("Episode Length", step, self.total_step)
                    self.writer.add_scalar("Epsilon", self.epsilon, self.total_step)
                    self.writer.add_scalar("Loss", self.loss, self.total_step)
                    self.writer.flush()
                    break

                if len(self.memory.buffer) > self.replay_start_size:
                    self.loss = self.learn()
                    if step % self.target_update_step == 0:
                        self.q_target_net.load_state_dict(
                            self.q_net.state_dict())

                step += 1
                self.total_step +=1
                del state
                state = self.arr_to_tensor(next_state)
            if epi % self.save_interval == 0 and epi != 0:
                torch.save(
                    {'policy_net': self.q_net, 'target_net': self.q_target_net, 'replay_memory': self.memory.buffer,
                     'optimizer': self.optimizer, 'epi_num': epi}, "model/pretrained_model.pth")

            if epi % self.print_interval == 0 and epi != 0:
                print("episode: {} / step: {:.2f} / reward: {:.3f}".format(epi, np.mean(self.epi_length),
                                                                           np.mean(self.rewards)))
                self.epi_length = []
                self.rewards = []

        torch.save(self.q_net, 'model/dqn_model_final.pth')
        self.env.close()
        self.writer.close()

    # 0 = forward
    # 1 = backward
    # 2 = left
    # 3 = right
    # 4 = forward + left
    # 5 = forward + right
    # 6 = backward + left
    # 7 = backward + right
    def change_action_to_continuous(self, action):
        arr = np.zeros(2)
        if action == 0 or action == 4 or action == 5:
            arr[1] = 1
        elif action == 1 or action == 6 or action == 7:
            arr[1] = -1
        if action == 2 or action == 4 or action == 6:
            arr[0] = 1
        elif action == 3 or action == 5 or action == 7:
            arr[0] = -1
        return torch.tensor(arr)

    def learn(self):
        # not starting learning until sufficient data is collected
        if len(self.memory.buffer) < self.batch_size:
            return

        minibatch = self.memory.sample()
        states, actions, rewardss, next_states, dones = torch.empty(len(minibatch))
        for i in range(self.batch_size):
            states = torch.cat(minibatch[i][0])
            actions = torch.cat(minibatch[i][1])
            rewardss = torch.cat(minibatch[i][2])
            next_states = torch.cat(minibatch[i][3])
            dones = torch.cat(minibatch[i][4])

        q_values = self.q_net(states).gather(1, actions)
        max_next_q = self.q_target_net(next_states).max(1)[0].unsqueeze(1).detach()
        target = rewardss + self.gamma * max_next_q

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        states = next_states
        return loss

    def arr_to_tensor(self, state):
        changed = np.expand_dims(np.transpose(state[0], (2,0,1)),axis=0)
        return torch.from_numpy(changed).to(self.device)

    def get_action(self, state):
        # epsilon-greedy
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.epsilon_max_frame
        rand = random.random()
        return_val = None
        if rand > self.epsilon:
            return torch.argmax(self.q_net(state))
        else:
            return torch.randint(0, self.number_of_actions, (1,), device=self.device)



if __name__ == "__main__":
    a = Agent()
    a.main()
