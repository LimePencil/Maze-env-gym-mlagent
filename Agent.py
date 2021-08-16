import torch
from ReplayBuffer import ReplayBuffer
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch.optim as optim
import time
import random
import torch.nn.functional as F
from DQN import DQN
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


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
        unity_env = UnityEnvironment(path_to_env, no_graphics=False)
        self.env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)

        self.writer = SummaryWriter('runs/maze_test_dqn_1')

        self.rewards = []
        self.epi_length = []

        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.epsilon_max_frame = 1000000

        buffer_size_limit = 20000  # need to be set so that it does not go over the memory limit
        self.number_of_episode = 10000
        self.target_update_step = 2500
        self.print_interval = 5
        self.save_interval = 5
        self.replay_start_size = 500
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

    # all the learning/training loop is in this function
    def main(self):
        for epi in range(self.number_of_episode):
            state = self.env.reset()
            state = self.arr_to_tensor(state)
            cumulative_reward = 0
            step = 0
            while True:
                # TODO: make sure to change multiplier of movement so that the player do not move too slowly
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(self.change_action_to_continuous(action))
                cumulative_reward += reward
                # if step < 20:
                #     y = next_state[0]*255
                #     z = y.astype(np.uint8)[:, :, 0]
                #     im = Image.fromarray(z, 'L')
                #     im.show()
                next_state = self.arr_to_tensor(next_state)
                if self.train_mode:
                    self.memory.push(state, action, torch.tensor([reward]).to(self.device), next_state)
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
                    self.writer.flush()
                    break
                # learning
                if len(self.memory.buffer) > self.replay_start_size:
                    self.loss = self.learn()
                    self.writer.add_scalar("Loss", self.loss, self.total_step)
                    if step % self.target_update_step == 0:
                        self.q_target_net.load_state_dict(
                            self.q_net.state_dict())

                step += 1
                self.total_step += 1
                del state
                state = next_state
                del next_state
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

    # since DQN only supports discrete actions, changing Box Action input to Continuous action is needed
    # 0 = forward
    # 1 = backward
    # 2 = left
    # 3 = right
    # 4 = forward + left
    # 5 = forward + right
    # 6 = backward + left
    # 7 = backward + right
    def change_action_to_continuous(self, action):
        arr = np.zeros((2,))
        if action == 0 or action == 4 or action == 5:
            arr[1] = 1
        elif action == 1 or action == 6 or action == 7:
            arr[1] = -1
        if action == 2 or action == 4 or action == 6:
            arr[0] = 1
        elif action == 3 or action == 5 or action == 7:
            arr[0] = -1
        return arr

    def learn(self):
        # not starting learning until sufficient data is collected
        if len(self.memory.buffer) < self.batch_size:
            return
        minibatch = self.memory.sample()
        batch = self.memory.Transition(*zip(*minibatch))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        q_values = self.q_net(state_batch).gather(1, action_batch)
        max_next_q = self.q_target_net(non_final_next_states).max(1)[0].unsqueeze(1).detach()
        target = reward_batch + self.gamma * max_next_q

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def arr_to_tensor(self, state):
        changed = np.expand_dims(np.transpose(state[0], (2, 0, 1)), axis=0)
        return torch.from_numpy(changed).to(self.device)

    def get_action(self, state):
        # epsilon-greedy
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.epsilon_max_frame
        rand = random.random()
        return_val = None
        if rand > self.epsilon:
            # this was the freaking bug that I searched for 2 hours ahhhhhhh
            return self.q_net(state).max(1)[1].view(1,)
        else:
            return torch.randint(0, self.number_of_actions, (1,), device=self.device)


if __name__ == "__main__":
    a = Agent()
    a.main()
