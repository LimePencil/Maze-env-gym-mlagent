import torch
import torch.nn as nn
import random

class DQN(nn.module):
    # creating neural network
    def __init__(self):
        # hyperparameters
        self.number_of_actions = 8
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.epsilon_max_frame = 1000000

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        # flatten the neur
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out
    def get_action(self, state):
        # epsilon-greedy
        out = self.forward(state)
        if self.epsilon >  self.final_epsilon:
            self.epsilon -= (self.initial_epsilon-self.final_epsilon)/self.epsilon_max_frame
        
        rand = random.random()

        if rand > self.epsilon:
            return torch.argmax(out).item()
        else:
            return random.randint(0,self.number_of_actions-1)







