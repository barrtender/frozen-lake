import numpy as np
import gymnasium as gym
from gymnasium import __version__
print(__version__)
import random
import time
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt
plt.ion()

from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)
        
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

e = Experience(2,3,1,4)

print(e)

class ReplayMemory():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay) -> None:
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)
            
            
class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        
        if rate > random.random():
            action = random.randrange(self.num_actions) # explore
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) # exploit
            
class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
        # Note: render_mode is moved to `make` as opposed to `render` as of 0.25.0.
        # I don't know if this is going to cause problems in the tutorial
        self.env = gym.make('CartPole-v1', render_mode="rgb_array").unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
        
    def reset(self):
        self.env.reset()
        self.current_screen = None
        
    def close(self):
        self.env.close()
        
    def render(self, mode="human"):
        return self.env.render()
    
    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self, action):
        # Updated for the most recent version
        # See docs here https://gymnasium.farama.org/api/env/#gymnasium.Env.step
        _, reward, terminated, truncated, _ = self.env.step(action.item())
        self.done = terminated or truncated
        return torch.tensor([reward], device=self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
        
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
     
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        
        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen
    
    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        
        # Use torchfvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40,90)),
            T.ToTensor()
        ])
        
        return resize(screen).unsqueeze(0).to(self.device) # add a batch dimension
    
device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
em.reset()
screen = em.render('rgb_array')

plt.figure()
plt.imshow(screen)
plt.title("Non-processed screen example")
plt.show()
plt.draw()
plt.pause(0.001)

input("enter to continue")

# Example start screen. Just black
screen = em.get_state()
# plt.figure()
plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
plt.title('Procressed screen example')
plt.show()
plt.draw()
plt.pause(0.001)

input("enter to continue")

for i in range(5):
    em.take_action(torch.tensor([1]))

# Example mid-way screen. Shows the progress by diffing the step with previous step
screen = em.get_state()
# plt.figure()
plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
plt.title('Procressed screen example')
plt.show()
plt.draw()
plt.pause(0.001)

input("enter to continue")

# Example done screen
em.done = True
screen = em.get_state()
# plt.figure()
plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
plt.title('Procressed screen example')
plt.show()
plt.draw()
plt.pause(0.001)

input("enter to continue")

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)
    # idk what this is.
    #if is_ipython: display.clear_output(wait=True)
    
def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
plot(np.random.rand(300), 100)
input("enter to continue")