import collections
import random


class ReplayBuffer():
    def __init__(self, batch_size, size_limit):
        self.buffer = collections.deque(maxlen=size_limit)
        self.batch_size = batch_size
        self.Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    def push(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample(self):
        return random.sample(self.buffer, self.batch_size)
