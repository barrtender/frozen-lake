import gymnasium as gym
from gymnasium import __version__
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as Axes
import matplotlib.figure as Figure
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print(__version__)
matplotlib.use('TkAgg')
plt.ion()

class SimpleDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)  # CartPole state is [position, velocity, angle, angular_velocity]
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))

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
        self.env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
        self.env.reset()
        self.done = False

    def reset(self):
        observation, _ = self.env.reset()  # Updated for new gym API
        self.last_observation = observation
        self.done = False

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render()

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action): 
        # Updated for the most recent version
        # See docs here https://gymnasium.farama.org/api/env/#gymnasium.Env.step
        observation, reward, terminated, truncated, _ = self.env.step(action.item())
        self.done = terminated or truncated
        self.last_observation = observation  # Store for next get_state() call
        if self.done:
            reward = -1
        return torch.tensor([reward], device=self.device)

    def get_state(self):
        # Instead of processing screen, just return the raw observation
        if hasattr(self, 'last_observation'):
            return torch.tensor(self.last_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            # First call, return zeros
            return torch.zeros(1, 4, device=self.device)

def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.next_state)
    t4 = torch.cat(batch.reward)    
    
    return (t1,t2,t3,t4)

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Return the predicted Q values from the policy net for the given state and action pair
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        
        # Instead of trying to detect terminal states from the screen,
        # just compute Q-values for all states and let the learning process handle it
        values = target_net(next_states).max(dim=1)[0].detach()
        return values

def show_updates(em):
    realScreen = em.render("rgb_array")
    realFig.set_data(realScreen)
    plt.pause(0.001)


def plot(values, moving_avg_period):
    ax2.clear()
    ax2.plot(values, "-b")
    moving_avg = get_moving_average(moving_avg_period, values)
    ax2.plot(moving_avg, "-r")
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])


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

# A couple example steps before training.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
em.reset()

# I couldn't figure out how to get the type hinting without making a new function
def getPlots() -> tuple[Figure.Figure, tuple[Axes.Axes, Axes.Axes]]:
    return plt.subplots(1, 2)
fig, (ax1, ax2) = getPlots()

# Example start screen with blank training plot
realScreen = em.render("rgb_array")
realFig = ax1.imshow(realScreen)
ax2.set_title("Training...")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Duration")
plt.show()
plt.draw()
plt.pause(0.001)
print("example start pole")
input("enter to continue")

# Example mid-way screen. Shows the progress of a few actions
for i in range(5):
    em.take_action(torch.tensor([1]))
    show_updates(em)
    plt.pause(0.001)

print("after a few steps, pole")
input("enter to continue")


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

policy_net = SimpleDQN().to(device=device)
target_net = SimpleDQN().to(device=device)
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
    
    print(f"\n=== Episode {episode} ===")
    episode_reward = 0
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
        
        # This was an idea to give it extra bonus points for lasting longer.
        # But it seems like it just confused the rewards. I'll have to think more here.
        # episode_reward = reward * timestep
        # if reward.item() <= 0:
        #     episode_reward = reward 
            
        next_state = em.get_state()
        
        # I don't need to watch every attempt, maybe just 1 in 30
        if episode % 30 == 0:
            show_updates(em)
            
        memory.push(Experience(state, action, next_state, reward))
        # It got really good so I need to stop it at some point or it'll never finish
        if em.done or timestep > 500:
            if timestep > 500:
                print("Got to the limit")
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
        state = next_state
        
        """
        5. Sample random batch from replay memory
        """
        if memory.can_provide_sample(batch_size):
            experience = memory.sample(batch_size)
            """
            6. Preprocess states from batch
            7. Pass batch of preprocessed states to policy network
            """
            states, actions, next_states, rewards = extract_tensors(experience)
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
            # Calculate the necessary updates to the weights and biases (the gradient).
            # This sets the .grad property on all the parameters in the network
            # Since the loss was passed the current_q_values which was passed the policy_net
            # the PyTorch "autograd system" can trace all the way back to the original policy_net, 
            # so it's still connected despite never being passed directly to loss
            loss.backward()
            # Apply updates to weights and biases
            # This applies the .grad property that was set by the loss.backward() call
            optimizer.step()
        
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

input("enter to continue")
em.close()

# TODO: https://www.youtube.com/watch?v=N23YrridnAc&list=PLccH6XYi5vIkyYoOZyDAz4hw8XSoOcW_K&index=6