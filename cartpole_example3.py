import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from random import random
from torch.distributions import Categorical



ENV = gym.make('CartPole-v1')
state = ENV.reset()

# Don't need a big neural network
class PolicyNN(nn.Module):
    def __init__(self):
        super(PolicyNN,self).__init__()
        self.fc = nn.Linear(ENV.observation_space.shape[0],ENV.action_space.n)

    def forward(self,x):
        x = self.fc(x)
        return F.softmax(x, dim=1)


    
def select_action(state,action_num,model):
    # random
    if action_num == 1:
        return 0 if random() < 0.5 else 1
    # simple
    if action_num == 2:
        return 0 if state[2] < 0 else 1
    # good
    if action_num == 3:
        return 0 if state[2] + state[3] < 0 else 1
    # policy
    if action_num == 4:
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = model(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    # policy best
    if action_num == 5:
        tate = torch.from_numpy(state).float().unsqueeze(0)
        probs = model(state)
        if probs[0][0] > probs[0][1]:
            return 0
        else:
            return 1

def train_simple(action_num, num_episodes=10*100):
    model = PolicyNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_steps = 500
    ts = []
    for episode in range(num_episodes):
        state = ENV.reset()
        probs = []
        for t in range(1, num_steps+1):
            if action_num != 4:
                action = select_action(state,action_num)
                state, _, done, _ = ENV.step(action)
                
            else:
                action, prob = select_action(state,action_num,model)
                probs.append(prob)
                state, _, done, _ = ENV.step(action)
            
            ENV.render()
            if done:
                break

        if action_num == 4:
            loss = 0
            for i, prob in enumerate(probs):
                loss += -1 * (t - i) * prob
            print(episode, t, loss.item())

            # Zeroing gradient before computing the gradeint
            optimizer.zero_grad()
            # Computing the gradient of current tensor
            loss.backward()
            # Updating optimizer
            optimizer.step()

        ts.append(t)
        print(f"episode #{episode} done")

        if len(ts) > 10 and sum(ts[-10:])/10.0 >= num_steps * 0.95:
            print('Stopping training, looks good...')
            return
        # print(f"ts: {ts}")
    # score = sum(ts) / (len(ts)*num_steps)
    # return score

if __name__ == '__main__':

    #1: random, #2: simple, #3: good, #4: policy
    # action_num = input("#1: random, #2: simple, #3: good, #4: policy \nWhich action do you want to run? ")
    # print(f"Running action number {action_num}")
    # action_num = int(action_num)
    action_num = 4
    train_simple(action_num, num_episodes=10*100)


    