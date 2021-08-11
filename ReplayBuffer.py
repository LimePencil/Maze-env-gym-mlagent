import collections
import random


class ReplayBuffer():
    def __init__(self, batch_size, size_limit):
        self.buffer = collections.deque(maxlen=size_limit,)
        self.batch_size = batch_size

    def push(self, data):
        self.buffer.append(data)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)
