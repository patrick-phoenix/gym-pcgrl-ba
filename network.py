import torch
import torch.nn as nn


class RewardNet(nn.Module):
    def __init__(self, input_size=196, hidden_size=64, hidden_layers = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)   # * unpacks the list, so Sequential gets all layers individually 


    def forward(self, x):
        return self.net(x)



def level_to_tensor(level):
    '''Takes a 2d level (with numbers as tiles) and flattens it to a tensor, 
    for using in a neural network.'''
    return torch.tensor(level.flatten()).float()


def train(net : RewardNet, winner_tensor : torch.Tensor, loser_tensor : torch.Tensor, 
          optimizer = None, learning_rate = 0.001, epochs = 100):
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    net.train()

    for i in range(epochs):
        optimizer.zero_grad()
        loss = calculate_loss(net, winner_tensor, loser_tensor)
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"loss in epoch {i}: {loss.item():.4f}")

    return optimizer


def calculate_loss(net, winner_tensor, loser_tensor):
    '''Calculates the cross-entropy-loss after the paper "Deep Reinforcement Learning from Human Preferences"
      by Christiano et al. 2017.  
      In this named paper it follows the Bradley-Terry model (Bradley and Terry, 1952) for 
      estimating score functions from pairwise preferences.  
      Here it is simplified to Pr(l_1 > l_2) = 1 / (1 + exp(r_2 - r_1)) = sigmoid(r_1 - r_2).  
      loss = - sum over dataset: µ(l_1) * log(Pr(l_1 > l_2))  + µ(l_2) * log(Pr(l_2 > l_1)) 
      with µ(l_1) and µ(l_2) being the labels for level 1 and 2.   
      µ(l_1) = 1, if level 1 is the winner and µ(l_1) = 0, if level 1 is the loser.
      '''

    r_w = net(winner_tensor)
    r_l = net(loser_tensor)
    log_prob =  nn.functional.logsigmoid(r_w - r_l)   # log-probability that winner is better than loser in reward-network
    loss = -log_prob.mean()     # Paper takes sum, but mean was chosen, because it is independent to size of dataset
    return loss


def accuracy(net, winner_tensor, loser_tensor):
    '''Calculates the accuracy of the RewardNet. 
    Returns a fractions of how many labels match with choice of RewardNet '''
    with torch.no_grad():
        r_w = net(winner_tensor)
        r_l = net(loser_tensor)

    level_count = len(winner_tensor)
    matches = 0

    for level in range(level_count):
        if r_w[level] > r_l[level]:
            matches += 1

    return matches / level_count


def load_reward_network(path):
    '''Takes a path, where the configurations and the weights for the reward network
      are saved and returns a reward network with these weights '''
    checkpoint = torch.load(path)
    net = RewardNet(**checkpoint["config"])
    net.load_state_dict(checkpoint["state_dict"])
    return net


def save_reward_network(net, path):
    '''Takes a reward network and path, where the configurations and weights for it
      will be saved.'''
    checkpoint = {
        "config": {
            "input_size": net.input_size,
            "hidden_size": net.hidden_size,
            "hidden_layers": net.hidden_layers,
        },
        "state_dict": net.state_dict(),
    }
    torch.save(checkpoint, path)




if __name__ == "__main__":
    net = RewardNet(hidden_layers=2)
    fake_levels = torch.zeros(5, 196)
    output = net(fake_levels)
    print(output.shape)   # Ziel: torch.Size([5, 1])