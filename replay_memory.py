import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):  
        self.buf.append(args)
    def sample(self, n):    
        return random.sample(self.buf, min(n, len(self.buf)))
    def __len__(self):      
        return len(self.buf)
