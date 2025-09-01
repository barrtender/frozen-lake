import numpy as np
import gymnasium as gym
from gymnasium import __version__

print(__version__)
import random
import time
import math
from IPython.display import clear_output
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.axes as Axes
import matplotlib.figure as Figure

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

        self.fc1 = nn.Linear(in_features=img_height * img_width * 3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))

e = Experience(2, 3, 1, 4)

print(e)


class ReplayMemory:
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


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay) -> None:
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(
            -1.0 * current_step * self.decay
        )


class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)  # explore
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)  # exploit


class CartPoleEnvManager:
    def __init__(self, device):
        self.device = device
        # Note: render_mode is moved to `make` as opposed to `render` as of 0.25.0.
        # I don't know if this is going to cause problems in the tutorial
        self.env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
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
        screen = self.render("rgb_array").transpose((2, 0, 1))
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
        resize = T.Compose([T.ToPILImage(), T.Resize((40, 90)), T.ToTensor()])

        return resize(screen).unsqueeze(0).to(self.device)  # add a batch dimension



def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    
    return (t1,t2,t3,t4)

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Return the predicted Q values from the policy net for the given state and action pair
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
em.reset()

# I couldn't figure out how to get the type hinting without making a new function
def getPlots() -> tuple[Figure.Figure, tuple[Axes.Axes, Axes.Axes]]:
    return plt.subplots(1, 2)
fig, (ax1, ax2) = getPlots()

# Example start screen with just black diff screen
realScreen = em.render("rgb_array")
realFig = ax1.imshow(realScreen)
screen = em.get_state()
# Need to add a `.cpu()` here because matplotlib needs cpu as a device not gpu
# Same in the other places we're about to draw things.
stateFig = ax2.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation="none")
plt.show()
plt.draw()
plt.pause(0.001)
print("example start pole with black screen")
input("enter to continue")

# Example mid-way screen. Shows the progress by diffing the step with previous step
for i in range(5):
    em.take_action(torch.tensor([1]))
    realScreen = em.render("rgb_array")
    realFig.set_data(realScreen)
    stateScreen = em.get_state()
    stateFig.set_data(stateScreen.squeeze(0).permute(1, 2, 0).cpu())
    plt.pause(0.001)

    # input("enter to continue")

print("after a few steps, pole and diff")
input("enter to continue")


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])
    # idk what this is.
    # if is_ipython: display.clear_output(wait=True)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = (
            values.unfold(dimension=0, size=period, step=1)
            .mean(dim=1)
            .flatten(start_dim=0)
        )
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()



# Now on to training. Here's what we're doing

"""
1. Initialize replay memory capacity
2. Initialize the policy network with random weights.
3. Clone the policy network and call it the taret network
4. For each episode:
  1. Initialize the starting state
  2. For each time step:
    1. Select an action. Via exploration or exploitation
    2. Execute selected action in an emulator
    3. Observae reward and next state
    4. Store experience in replay memory
    5. Sample random batch from replay memory
    6. Preprocess states from batch
    7. Pass batch of prerprocessed states to policy network
    8. Calculate loss between output Q-values and target Q-values
      - Requires a pass to the target network for the next state
    9. Gradient descent updates weights in the policy network to minimize loss
      - After x time steps, weights in the target network are updated to the weights in the policy network
"""


batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
learning_rate = 0.001
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)
"""
1. Initialize replay memory capacity
2. Initialize the policy network with random weights.
3. Clone the policy network and call it the taret network
"""
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device=device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device=device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)

episode_durations = []
"""
4. For each episode:
  1. Initialize the starting state
"""
for episode in range(num_episodes):
    
    em.reset()
    state = em.get_state()
    
    """
    2. For each time step:
        1. Select an action. Via exploration or exploitation
        2. Execute selected action in an emulator
        3. Observe reward and next state
        4. Store experience in replay memory
        5. Sample random batch from replay memory
        6. Preprocess states from batch
        7. Pass batch of prerprocessed states to policy network
        8. Calculate loss between output Q-values and target Q-values
        - Requires a pass to the target network for the next state
        9. Gradient descent updates weights in the policy network to minimize loss
        - After x time steps, weights in the target network are updated to the weights in the policy network
    """
    for timestep in count():
        """
        1. Select an action. Via exploration or exploitation
        2. Execute selected action in an emulator
        3. Observe reward and next state
        4. Store experience in replay memory
        """
        action = agent.select_action(state, policy_net=policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        
        """
        5. Sample random batch from replay memory
        """
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            """
            6. Preprocess states from batch
            7. Pass batch of preprocessed states to policy network
            """
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            
            """
            8. Calculate loss between output Q-values and target Q-values
            - Requires a pass to the target network for the next state
            9. Gradient descent updates weights in the policy network to minimize loss
              - After x time steps, weights in the target network are updated to the weights in the policy network
            """
            # q(s,a) = E[R(t+1)+Gamma*maxQ(s',a')]
            target_q_values = rewards + (next_q_values * gamma)
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            # Zero-out gradients to prevent accumulation from previous backprop runs
            optimizer.zero_grad()
            # Calculate the necessary updates to the weights and biases (the gradient)
            loss.backward()
            # Apply updates to weights and biases
            optimizer.step()
            
        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
        
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

input("enter to continue")
em.close()

# TODO: https://www.youtube.com/watch?v=N23YrridnAc&list=PLccH6XYi5vIkyYoOZyDAz4hw8XSoOcW_K&index=6