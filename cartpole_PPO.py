# https://www.youtube.com/watch?v=l1CZQWBkdcY
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Creating the environment
ENV = gym.make("CartPole-v1")

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        # Parameters
        self.data = []
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps = 0.1
        self.K = 3 #epoch
        self.learning_rate = 0.001
        self.hidden_layer = 64

        # Setting up neural network 
        self.fc1 = nn.Linear(ENV.observation_space.shape[0],self.hidden_layer)
        self.fc_pi = nn.Linear(self.hidden_layer, ENV.action_space.n)
        self.fc_v = nn.Linear(self.hidden_layer, 1) 
        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)

    def pi(self,x, softmax_dim = 0): # Sample is only one, so the dim of softmax = 0. 0 if simulattion 1 if training
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        # softmax calculates the probability
        prob = F.softmax(x, dim = softmax_dim)
        return prob
    
    def v(self,x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, item):
        self.data.append(item)

    # Putting data into a batch
    def make_batch(self):
        s_list, a_list, r_list, next_s_list, prob_a_list, done_list = [], [], [], [], [], []

        # Getting data from self.data, then appending into a list
        # print(f"data set: {self.data}")
        for item in self.data:
            s,a,r,next_s,prob_a, done= item
            # s is a numpy array
            s_list.append(s) 
            # a and r are integers, so bracketting to fit into the same format (list)
            a_list.append([a])
            r_list.append([r])
            next_s_list.append(next_s)
            prob_a_list.append([prob_a])
            done_mask = 0 if done else 1
            done_list.append([done_mask])

        # Converting the dataset into a torch
        s,a,r,next_s,done_mask, prob_a = torch.tensor(s_list, dtype = torch.float),torch.tensor(a_list), torch.tensor(r_list), torch.tensor(next_s_list, dtype = torch.float),torch.tensor(done_list, dtype = torch.float), torch.tensor(prob_a_list)

        # Resetting self.data
        self.data =[]

        return s,a,r,next_s, done_mask, prob_a 

    def train(self):
        # Calling dataset from training
        s,a,r,next_s,done_mask, prob_a = self.make_batch()

        # Running for K number of epoch
        for i in range(self.K):
            ## GAE (Generalized Advantage Estimation) calculation
            # Calculating delta, using v (network): prediction, but doing this as a whole makes the calculation faster 
            td_target = r + self.gamma * self.v(next_s) * done_mask
            delta = td_target - self.v(s) # is a tensor
            # Change to numpy for later calculation
            delta = delta.detach().numpy()

            # Advantage list
            advantage_lst = []
            # Initial advantage is a zero
            advantage = 0.0
            # Calculating GAE using delta calculated from above. Not calling the network
            # Calculating from the back to be less computationally expensive
            # If done from the beginning, then too many things to add (t to T-1)
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            # Reversing the list
            advantage_lst.reverse()
            # Making the dataset into a tensor
            advantage = torch.tensor(advantage_lst, dtype = torch.float)

            self.loss_calc(s,a,advantage,prob_a,td_target)


    def loss_calc(self,s,a,advantage,prob_a,td_target):
        "Calculating Loss function"
        # Calculating the probability from a network
        pi = self.pi(s,softmax_dim=1)
        # 실제 했던 액션중의 확률 = 최신 policy를 이용한 new probability
        pi_a = pi.gather(1,a)
        # prob_a is the probability of the action performed when collecting experiment
        ratio = torch.exp(torch.log(pi_a)-torch.log(prob_a)) # a/b == exp(log(a)-log(b ))

        # PPO Clipped Loss Calculation - from PPO paper
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

        # Policy loss (needs to be maximized) + Value Loss (needs to be minimized)
        # detaching because td_target is a tree but we only want one value (want to disregard other trees)
        loss = -torch.min(surr1,surr2) + F.smooth_l1_loss(td_target.detach(),self.v(s))
        
        # Zeroing gradient before computing the gradeint
        self.optimizer.zero_grad()
        # Computing the gradient of current tensor
        loss.mean().backward()
        # Updating optimizer
        self.optimizer.step()
        

def main():
    # Using PPO alogrithm
    model = PPO()
    # Parameters
    gamma = 0.99
    # 몇 time step 동안 data를 모을지, Run policy for T timesteps, estimate advantage function at all timesteps
    T = 20 
    # Initial Score
    score = 0.0
    print_interval = T
    # Number of episodes
    episodes = 10000

    for n_epi in range(episodes):
        # Getting initial state
        state = ENV.reset()
        done = False
        while not done:
            # Obtaining data for 20 time steps, to obtain experiences
            for t in range(T):
                # Obtaining the probability
                prob = model.pi(torch.from_numpy(state).float())
                # Make the probability a Categorical variable
                m = Categorical(prob)
                # Sampling an action based on the probability
                a = m.sample().item()

                # Next Step with given action
                next_state, reward, done, info = ENV.step(a)
                # ENV.render()
                # Actual probability of the action taken. Used for ratio calculation later on
                real_prob = prob[a].item()
                # Saving the data for training later
                model.put_data((state, a, reward/100.0, next_state, real_prob , done))
                state = next_state

                score += reward
                if done:
                    break

            # Train based on the experiences collected from above
            model.train()
        
        if n_epi%print_interval == 0 and n_epi != 0:
            print(f"# of episode: {n_epi}, avg score: {score/print_interval}")
            if score/print_interval > 480:
                "Good enough"
                break
            score = 0.0
    ENV.close()


if __name__ == "__main__":
    main()