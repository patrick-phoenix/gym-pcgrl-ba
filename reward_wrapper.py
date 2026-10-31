import gym
import torch
import network
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
from gym_pcgrl.wrappers import CroppedImagePCGRLWrapper


class RewardWrapper(gym.Wrapper):
    '''Wraps around the gym-pcgrl environments, so the learned Reward Network 
    can give the reward to the agent instead of the original code. '''

    def __init__(self, env, reward_network):
            super().__init__(env = env)
            self.net = reward_network
            self.net.eval()


    def _score(self, level) -> float:
        '''Takes a level as a matrix and gives it a score from the reward network,'''
        with torch.no_grad():
                level_tensor = network.level_to_tensor(level)
                score = self.net(level_tensor).item()
        return score


    def step(self, action):
        # Copy old level
        raw_env : PcgrlEnv = self.env.unwrapped
        old_level = raw_env._rep._map.copy()

        obs, old_reward, done, info = self.env.step(action)
        info["original_reward"] = old_reward

        # Copy new level
        new_level = raw_env._rep._map.copy()

        # Calculate new reward
        old_lvl_score = self._score(old_level)
        new_lvl_score = self._score(new_level)
        new_reward = new_lvl_score - old_lvl_score

        return obs, new_reward, done, info


def test():
    env = CroppedImagePCGRLWrapper("binary-narrow-v0", 28)
    net = network.RewardNet()
    wrapper = RewardWrapper(env, net)
    wrapper.reset()

    for i in range(10):
        random_action = wrapper.action_space.sample()
        obs, reward, done, info = wrapper.step(random_action)
        print(f"---------- Step {i} ----------")
        print("Action taken: ", random_action)
        print("Original reward: ", info["original_reward"])
        print("new reward: ", reward)

if __name__ == "__main__":
    test()