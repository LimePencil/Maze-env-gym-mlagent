import torch
from ReplayBuffer import ReplayBuffer
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch.optim as optim
from PIL import Image
import time
import math
import random
import torch.nn.functional as F
from DQN import DQN


class Agent:
    def __init__(self):
        self.train_mode = True
        self.load = False
        self.path_to_save_file = ""
        self.number_of_actions = 8
        self.q_net = DQN(self.number_of_actions)
        self.q_target_net = DQN(self.number_of_actions)
        self.learning_rate = 0.00025
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.loss = None

        path_to_env = "envs/Ml-agent-with-gym"
        unity_env = UnityEnvironment(path_to_env)
        self.env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)

        self.rewards = []
        self.epi_length = []

        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.epsilon_max_frame = 1000000

        buffer_size_limit = 1000000
        self.number_of_episode = 1000000
        self.number_of_steps_per_epi = 20000
        self.target_update_step = 10000
        self.print_interval = 100
        self.save_interval = 500
        self.replay_start_size = 50000
        self.batch_size = 32
        self.gamma = 0.99
        self.memory = ReplayBuffer(self.batch_size, buffer_size_limit).buffer

        if self.load:
            checkpoint = torch.load(self.path_to_save_file)
            self.q_net = checkpoint['policy_net']
            self.q_target_net = checkpoint['target_net']
            self.memory = checkpoint['replay_memory']
            self.optimizer = checkpoint['optimizer']
    def state_conversion(self,state):
        #using only obervation data
        tensor = torch.from_numpy(state[0])
        return tensor
    def main(self):
        for epi in range(self.number_of_episode):
            state = self.env.reset()
            cumulative_reward = 0
            step = 0
            while True:
                action = self.get_action(self.state_conversion(state))
                next_state, reward, done, info = self.env.step(action)
                cumulative_reward += reward
                # if t%1000==0:
                #     im = Image.fromarray(observation[0][:, :, 0], 'L')
                #     im.show()
                if self.train_mode:
                    self.memory.push(self.state_conversion(state), action, reward, next_state, done)
                else:
                    self.env.render()
                    time.sleep(0.01)

                if done or step < (self.number_of_steps_per_epi - 1):
                    rewards = np.append(rewards, cumulative_reward)
                    epi_length = np.append(epi_length, step)
                    break

                if len(self.memory) > self.replay_start_size:
                    self.loss = self.learn()
                    if step % self.target_update_step == 0:
                        self.q_target_net.load_state_dict(
                            self.q_net.state_dict())

                step += 1
                state = next_state

            if epi % self.save_interval == 0 and epi != 0:
                torch.save({'policy_net': self.q_net, 'target_net': self.q_target_net, 'replay_memory': self.memory,
                            'optimizer': self.optimizer, 'epi_num': epi}, "model/pretrained_model" + str(epi) + ".pth")

            if epi % self.print_interval == 0 and epi != 0:
                print("episode: {} / step: {:.2f} / reward: {:.3f}".format(epi,
                                                                           np.mean(epi_length), np.mean(rewards)))
                epi_length = []
                rewards = []

        torch.save(self.q_net, 'model/dqn_model_final.pth')
        self.env.close()

    def learn(self):
        # not starting learning until sufficient data is collected
        if len(self.memory) < self.batch_size:
            return

        minibatch = self.memory.sample()
        states, actions, rewards, next_states, dones = torch.empty(size = self.batch_size)
        for i in range(self.batch_size):
            states = torch.cat(minibatch[i][0])
            actions = torch.cat(minibatch[i][1])
            rewards = torch.cat(minibatch[i][2])
            next_states = torch.cat(minibatch[i][3])
            dones = torch.cat(minibatch[i][4])

        q_values = self.q_net(states).gather(1, actions)
        max_next_q = self.q_target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + self.gamma * max_next_q

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        states = next_states
        return loss

    def get_action(self, state):
        # epsilon-greedy
        out = self.q_net(state)
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.epsilon_max_frame

        rand = random.random()

        if rand > self.epsilon:
            return torch.argmax(out).item()
        else:
            return random.randint(0, self.number_of_actions - 1)


if __name__ == "__main__":
    a = Agent()
    a.main()