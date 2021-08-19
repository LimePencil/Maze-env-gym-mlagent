import collections
import random


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
# buffer to save data for learning
class ReplayBuffer():
    def __init__(self, batch_size, size_limit):
        self.buffer = collections.deque(maxlen=size_limit)
        self.batch_size = batch_size

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self):
        return random.sample(self.buffer, self.batch_size)
