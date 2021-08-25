import os
import collections
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
        # using GPU if possible
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        # new environment from .exe file
        path_to_env = None
        if os.name == "nt":
            path_to_env = os.path.join("envs","windows", "Ml-agent-with-gym")
        elif os.name == "posix":
            path_to_env = os.path.join("envs","linux", "Ml-agent-with-gym.x86_64")
        unity_env = UnityEnvironment(path_to_env, no_graphics=False)
        self.env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)

        # tensorboard integration and summary writing
        self.writer = SummaryWriter('runs/maze_test_dqn_1')
        self.print_interval = 1

        # list for printing summary to terminal
        self.rewards = []
        self.epi_length = []

        # epsilon values
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.epsilon_max_frame = 1000000

        # variables for learning
        self.number_of_actions = 8
        self.learning_rate = 0.00025
        self.number_of_episode = 10000
        self.target_update_step = 10
        self.gamma = 0.99
        self.total_step = 0
        self.epi = 0

        # replay memory variables
        self.Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.replay_start_size = 5000
        self.batch_size = 32
        buffer_size_limit = 100000  # need to be set so that it does not go over the memory limit

        # save variables
        self.load = False
        self.path_to_save_file = os.path.join("model","pretrained_model.pth")
        self.save_interval = 10

        # three most important things
        self.q_net = DQN(self.number_of_actions).to(self.device)
        self.q_target_net = DQN(self.number_of_actions).to(self.device)
        self.memory = ReplayBuffer(self.batch_size, buffer_size_limit)

        # uses adam optimizer and huber loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.loss = None

        # time recording
        self.start_time = 0

        # loading from save file
        if self.load:
            checkpoint = torch.load(self.path_to_save_file)
            self.q_net.load_state_dict(checkpoint['policy_net'])
            self.q_target_net.load_state_dict(checkpoint['target_net'])
            self.memory.buffer = checkpoint['replay_memory']
            self.optimizer = checkpoint['optimizer']
            self.epi = checkpoint['epi_num']
            self.epsilon = checkpoint['epsilon']

    # all the learning/training loop is in this function
    def main(self):
        while self.epi < self.number_of_episode:
            # get state when reset
            state = self.np_to_tensor(self.env.reset())
            cumulative_reward = 0
            step = 0
            self.start_time = time.time()
            while True:
                action = self.get_action(state)

                next_state, reward, done, info = self.env.step(self.change_action_to_continuous(action))

                cumulative_reward += reward
                next_state = self.np_to_tensor(next_state)

                # push to replay memory
                self.memory.push(state, action, torch.tensor([reward]).to(self.device), next_state)

                # when it goes past max_step or reaches goal
                if done:
                    self.rewards.append(cumulative_reward)
                    self.epi_length.append(step)
                    # writing to tensorboard
                    self.writer.add_scalar("Rewards", cumulative_reward, self.total_step)
                    self.writer.add_scalar("Episode Length", step, self.total_step)
                    self.writer.add_scalar("Epsilon", self.epsilon, self.total_step)
                    self.writer.flush()
                    break

                # learning if possible
                if len(self.memory.buffer) > self.replay_start_size:
                    self.loss = self.learn()
                    self.writer.add_scalar("Loss", self.loss, self.total_step)
                    # updates target network every few steps
                    if step % self.target_update_step == 0:
                        self.q_target_net.load_state_dict(
                            self.q_net.state_dict())

                step += 1
                self.total_step += 1
                del state
                state = next_state
                del next_state
            self.checkpoint_save()
            self.print_summary()
            self.epi += 1

        # saving model in the end
        torch.save(self.q_net, os.path.join("model","dqn_model_final.pth"))
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

    # deep Q learning with replay memory
    def learn(self):
        # not starting learning until sufficient data is collected
        if len(self.memory.buffer) < self.batch_size:
            return

        # get samples for learning
        mini_batch = self.memory.sample()

        # pack it to individual tensor
        batch = self.Transition(*zip(*mini_batch))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)

        # get q value from policy network
        q_values = self.q_net(state_batch).gather(1, action_batch)

        # get q value from target network using next state
        max_next_q = self.q_target_net(non_final_next_states).max(1)[0].unsqueeze(1).detach()

        target = reward_batch + self.gamma * max_next_q

        # loss is square if >1 absolute value if <=1
        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        # backpropagation
        loss.backward()
        self.optimizer.step()
        return loss

    # getting action based on epsilon-greedy
    def get_action(self, state):
        # epsilon-greedy
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.epsilon_max_frame
        rand = random.random()
        if rand > self.epsilon:
            return self.q_net(state).max(1)[1].view(1, )
        else:
            return torch.randint(0, self.number_of_actions, (1,), device=self.device)

    # changing numpy array to image for verification
    def to_image(self, state):
        y = state * 255
        z = y.astype(np.uint8)[:, :, 0]
        im = Image.fromarray(z, 'L')
        im.show()

    # numpy array to tensor
    def np_to_tensor(self, state):
        changed = np.expand_dims(np.transpose(state[0], (2, 0, 1)), axis=0)
        return torch.from_numpy(changed).to(self.device)

    # saves every interval
    def checkpoint_save(self):
        if self.epi % self.save_interval == 0 and self.epi != 0:
            torch.save(
                {'policy_net': self.q_net.state_dict(), 'target_net': self.q_target_net.state_dict(),
                 'replay_memory': self.memory.buffer,
                 'optimizer': self.optimizer, 'epi_num': self.epi, 'epsilon':self.epsilon}, self.path_to_save_file)

    # printing summary in the terminal
    def print_summary(self):
        if self.epi % self.print_interval == 0 and self.epi != 0:
            print("episode: {} / step: {:.2f} / reward: {:.3f} / time: {:.2f}s".format(self.epi, np.mean(self.epi_length),
                                                                       np.mean(self.rewards),time.time()-self.start_time))
            self.epi_length = []
            self.rewards = []


if __name__ == "__main__":
    a = Agent()
    a.main()
