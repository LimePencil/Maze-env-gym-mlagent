import torch
from torch.optim import optimizer
from DQN import DQN
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


class agent():
    def main(self):
        self.q_net = DQN()
        self.q_target_net = DQN()
        self.optimizer = optim.Adam(self.q_net.parameters())

        path_to_env = "envs/Ml-agent-with-gym"
        unity_env = UnityEnvironment(path_to_env)
        env = UnityToGymWrapper(unity_env, uint8_visual=True,allow_multiple_obs =True)
        
        train_mode = True

        buffer_size_limit = 1000000
        number_of_episode = 1000000
        number_of_steps_per_epi = 20000
        target_update_step = 10000
        print_interval = 100
        save_interval = 500
        self.replay_start_size = 50000
        self.batch_size = 32
        self.gamma = 0.99
        self.memory = ReplayBuffer(self.batch_size, buffer_size_limit).buffer

        rewards = []
        epi_length = []

        for epi in range(number_of_episode):
            state = env.reset()
            cumulative_reward = 0
            step = 0
            while True:
                action = self.q_net.get_action(state)
                next_state, reward, done, info = env.step(action)
                cumulative_reward += reward
                # if t%1000==0:
                #     im = Image.fromarray(observation[0][:, :, 0], 'L')
                #     im.show()
                if train_mode:
                    self.memory.push(state,action,reward,next_state,done)
                else:
                    time.sleep(0.01)
                
                if done or step < (number_of_steps_per_epi - 1):
                    rewards.append(cumulative_reward)
                    epi_length.append(step)
                    break

                if len(self.memory) > self.replay_start_size:
                    self.loss = self.learn()
                    if step % target_update_step == 0:
                        self.q_target_net.load_state_dict(self.q_net.state_dict())
                
                step += 1
                state = next_state
            if epi % save_interval == 0 and epi !=0:
                torch.save(self.q_net, "trained_model" + str(epi) + ".pth")
            if epi % print_interval == 0 and epi != 0:
                print("episode: {} / step: {:.2f} / reward: {:.3f}".format(epi,np.mean(epi_length),np.mean(rewards)))
        env.close()
    def learn(self):
        # not starting learning until sufficient data is collected
        if len(self.memory) < self.batch_size:
            return

        minibatch = self.memory.sample()
        states, actions, rewards, next_states, dones = ()
        for i in range(self.batch_size):
            states = torch.cat(minibatch[i][0])
            actions = torch.cat(minibatch[i][1])
            rewards = torch.cat(minibatch[i][2])
            next_states = torch.cat(minibatch[i][3])
            dones = torch.cat(minibatch[i][4])

        q_values = self.q_net(states).gather(1,actions)
        max_next_q = self.q_target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + self.gamma*max_next_q
        loss = F.smooth_l1_loss(q_values,target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        states = next_states
        return loss
        
if __name__ == "__main__":
    agent.main()