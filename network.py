import torch
import torch.nn as nn


class RewardNet(nn.Module):
    def __init__(self, input_size=196, hidden_size=64, hidden_layers = 1):
        super().__init__()
        
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)   # * unpacks the list, so Sequential gets all layers individually 


    def forward(self, x):
        return self.net(x)



def extract_training_data(list_winner_loser : list):
    '''Takes a List of pairs of matrices that represent a generated level.  
    The first represents the winner and the second, the loser (the worse level of them both).  
        
    Returns two tensors, with flattend levels. The reulting shape is (number of pairs x level size).  
    The first tensor represents the winners and the second tensor, the losers.'''

    level_pairs = len(list_winner_loser)
    if level_pairs == 0:
        raise ValueError("no pairs")

    level_size = list_winner_loser[0][0].size

    winners = torch.zeros(level_pairs, level_size)
    losers = torch.zeros(level_pairs, level_size)

    for index, (w, l) in enumerate(list_winner_loser):
        winners[index] = torch.tensor(w.flatten())
        losers[index] = torch.tensor(l.flatten())

    return winners, losers


def train(net : RewardNet, winner_tensor : torch.Tensor, loser_tensor : torch.Tensor, 
          optimizer = None, learning_rate = 0.001, epochs = 100):
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
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


def evaluate(net, winner_tensor, loser_tensor):
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




if __name__ == "__main__":
    net = RewardNet(hidden_layers=2)
    fake_levels = torch.zeros(5, 196)
    output = net(fake_levels)
    print(output.shape)   # Ziel: torch.Size([5, 1])