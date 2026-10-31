import numpy as np
import json

from gym_pcgrl.envs.helper import get_string_map
import network
from network import RewardNet


OUT_FILE = "pairs.jsonl"


def get_dict_from_files(path = OUT_FILE):
    '''Takes the path to a file, where it reads every line and converts them into an array of python objects.'''
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        if line.strip():
            data.append(json.loads(s = line))
    return data


def get_winner_and_matrix(data, label_selection = "LLM"):
    '''Takes a list of dictionarys (data) which include LLM and criteria Labels, as well as the level matrices.  
    The label_selection can be either "LLM" or "Code". This determines from which label the winner/loser is chosen.  
    Returns a list of tupels of matrices, where the first is the winner and the second is the loser. If there is no
    winner or looser, this level pair is not included.'''

    if label_selection == "LLM":
        chosen_label = "LLM_label"
    elif label_selection == "Code":
        chosen_label = "criteria_label"
    else:
        raise ValueError(f"label_selection must be 'LLM' or 'Code', got {label_selection}")

    winners_and_losers = []
    for entry in data:
        label = entry[chosen_label]
        if label == "A":
            winner = np.array(entry["level_A"])
            loser = np.array(entry["level_B"])
            winners_and_losers.append((winner, loser))
        elif label == "B":
            winner = np.array(entry["level_B"])
            loser = np.array(entry["level_A"])
            winners_and_losers.append((winner, loser))

    return winners_and_losers


def extract_training_data(list_winner_loser : list):
    '''Takes a List of pairs of matrices that represent a generated level.  
    The first represents the winner and the second, the loser (the worse level of them both).  
        
    Returns two tensors, with flattened levels. The resulting shape is (number of pairs x level size).  
    The first tensor represents the winners and the second tensor, the losers.'''

    level_pairs = len(list_winner_loser)
    if level_pairs == 0:
        raise ValueError("no pairs")

    level_size = list_winner_loser[0][0].size

    winners = torch.zeros(level_pairs, level_size)
    losers = torch.zeros(level_pairs, level_size)

    for index, (w, l) in enumerate(list_winner_loser):
        winners[index] = level_to_tensor(w)
        losers[index] = level_to_tensor(l)

    return winners, losers



def train_save_reward_network(reward_network = None,
                            data_path = OUT_FILE, 
                            label_filter = "LLM", 
                            save_path = None,
                            optimizer = None, 
                            learning_rate = 0.001, 
                            epochs = 100):
    '''Loads a json with levels as training pairs and trains the reward network.  
    Afterwards it saves the weights.   
      
    Parameters:  
    reward_network: The Reward_Network to train. If = None, then a new one is created.
    data_path: The path to the json that includes level pairs and labels that say which
                 one is the winner/loser. Default is OUT_FILE.  
    label_filter: The label choice to determine the winner and loser.
                The Choice is between "LLM" and the original "Code".  
    save_path: The path, where the trained weights of the reward network, will be saved.
                If equal to None (default), the network won't get saved. 
    optimizer: The optimizer for the training. If = None (default), a new Adam optimizer 
                is created with the given learning rate.
    learning_rate: The learning rate for a new optimizer. Is ignored if an optimizer 
                is given. Default is 0.001. 
    epochs: The number of training iterations over the whole dataset. Default is 100.

    Returns the trained reward network and optimizer.'''
    if reward_network is None:
        reward_network = RewardNet()

    # Get Training Data
    training_data = get_dict_from_files(data_path)
    train_pairs = get_winner_and_matrix(training_data, label_filter)
    winners, losers = extract_training_data(train_pairs)

    # Training
    optimizer = network.train(reward_network, winners, losers, optimizer, learning_rate, epochs)

    if save_path is not None:
        network.save_reward_network(reward_network, save_path)
    return reward_network, optimizer


def evaluate_reward_network(reward_network, data_path, label_filter = "Code"):
    '''Loads a json with level pairs and calculates how often the reward network 
    agrees with the labels.  
      
    Parameters:  
    reward_network: The trained reward network to evaluate.  
    data_path: The path to the json that includes level pairs and labels that say which 
                one is the winner/loser. Should contain different pairs than the training data.  
    label_filter: The label choice to determine the winner and loser.
                The Choice is between "LLM" and the original "Code". Default is "Code".  

    Returns the accuracy as a fraction between 0 and 1.'''
    # Get test data
    test_data = get_dict_from_files(data_path)
    test_pairs = get_winner_and_matrix(test_data, label_filter)
    test_winners, test_losers = extract_training_data(test_pairs)

    # Calculate accuracy of reward network
    accuracy = network.accuracy(reward_network, test_winners, test_losers)
    return accuracy



def test():
    # Train and save
    net, optimizer = train_save_reward_network(data_path="pairs_none.jsonl",
                                                label_filter="Code",
                                                save_path="reward_net.pt")
    print("Accuracy before saving: ", evaluate_reward_network(net, "pairs_medium.jsonl"))

    # Load and evaluate again
    loaded_net = network.load_reward_network("reward_net.pt")
    print("Accuracy after loading: ", evaluate_reward_network(loaded_net, "pairs_medium.jsonl"))


if __name__ == "__main__":
    test()