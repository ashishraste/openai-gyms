import gym
from numpy import mean, argmax, argmin, zeros, \
    max as npmax, random, save as npsave, load as npload
from math import log10, radians
from collections import deque
from logging import getLogger, INFO, DEBUG, info, debug, error
from tqdm import tqdm
from tqdm_logger import TqdmLoggingHandler


class QCartPoleAgent():
    def __init__(self, state_buckets=(1,1,6,12), num_episodes=1000, min_alpha=0.1,\
         gamma=1, min_epsilon=0.1, min_successful_runs=195, episode_factor=25, verbose=False, saved_q_table=None):

        self.env = gym.make('CartPole-v0')
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], radians(50)]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -radians(50)]

        # Input transformation params
        self.state_buckets = state_buckets

        # Hyperparameters
        self.gamma = gamma
        self.episode_factor = episode_factor # Controls the learning and exploration rate over episodes. Starts with high values 
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon

        # If saved-artefact of Q-table is available from training, then load it for re-use
        if saved_q_table is not None:
            self.Q = saved_q_table
        else:
            # Initialize Q matrix, which maps from state-space to action-space
            self.Q = zeros(self.state_buckets + (self.env.action_space.n,))                    

        # Training variables
        self.num_episodes = num_episodes
        self.min_successful_runs = min_successful_runs  # Minimum number of successful runs required to call an episode successful
        
        self.verbose = verbose

    def discretize_state(self, obs):
        ''' Discretizes state variables, excluding cart-position and cart-velocity, to reduce the dimensionality of state-space. '''
        scaling_factors = [(obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i]) \
            for i in range(len(obs))]
        new_obs = [int(round((self.state_buckets[i] - 1) * scaling_factors[i])) \
            for i in range(len(obs))]
        # Clip transformed observation inputs to fall within bucket bounds
        new_obs = [min((self.state_buckets[i] - 1, max(0, new_obs[i]))) \
            for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        ''' Selects an action with an epsilon-greedy policy i.e. with a probability of epsilon we pick an unexplored action.'''
        return self.env.action_space.sample() if (random.random() <= epsilon) else argmax(self.Q[state])

    def update_q_value(self, old_state, action, reward, new_state, alpha):
        self.Q[old_state][action] += alpha * (reward + self.gamma * npmax(self.Q[new_state]) - self.Q[old_state][action])
        
    def get_epsilon(self, episode_num):
        ''' Returns epsilon value which attributes to the interest in exploration, and 
        which is usually high in the early episodes i.e. early stages of learning.'''
        return max(self.min_epsilon, min(1, 1. - log10((episode_num+1) / self.episode_factor)))

    def get_alpha(self, episode_num):
        return max(self.min_alpha, min(1, 1. - log10((episode_num+1) / self.episode_factor)))

    def run(self, mode='train', successful_episode_index=0, disable_progress_bar=False):
        cumulative_reward = deque(maxlen=100)
        if mode == 'test':
            if 0 == successful_episode_index:
                error('Run in test mode requires the episode index from which last training succeeded. If not available, please run in \'train\' mode. Quitting for now ...')
                exit(-1)
            else: successful_episode = successful_episode_index
            alpha = self.get_alpha(successful_episode)
            epsilon = self.get_epsilon(successful_episode)

        for ep in tqdm(range(self.num_episodes), desc="Episode Run", disable=disable_progress_bar):
            if mode == 'train':
                alpha = self.get_alpha(ep)
                epsilon = self.get_epsilon(ep)

            episode_done = False
            time_step = 0
            cur_state = self.discretize_state(self.env.reset())

            while not episode_done:
                self.env.render()
                action = self.choose_action(cur_state, epsilon)
                obs, reward, episode_done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                self.update_q_value(cur_state, action, reward, new_state, alpha)
                cur_state = new_state
                time_step += 1

            # Successful episodes run for more than 200 time steps.
            debug('Score for episode {} is {}'.format(ep, time_step))
            cumulative_reward.append(time_step)
            
            # Calculate mean cumulative reward over last 100 episodes.
            mean_cumulative_reward = mean(cumulative_reward)
            if mean_cumulative_reward >= self.min_successful_runs and ep >= 100:
                if self.verbose: info('Solved after {} episodes. {} trials were taken.'.format(ep+1, ep-99))
                return ep - 99
            if ep >= 100 and ep % 100 == 0:
                debug('Episode {}: Mean time-steps before termination over last 100 episodes was {}.'.format(ep+1, mean_cumulative_reward))

        if self.verbose: info('Couldn\'t solve after {} episodes.'.format(ep+1))
        return ep + 1
    
    def save_trained_artefacts(self, episodes_taken, filename='cartpole_artefacts'):
        ''' Saves Q-table and episode index from which successful runs started into a numpy file. 
        Episode index is required since we set adaptive learning-rate (alpha) and exploration probability (epsilon) based on it.
        '''
        artefacts_to_save = {'Q':self.Q, 'successful_episode_index': episodes_taken} 
        npsave(filename, artefacts_to_save)


def load_trained_artefacts(filename='cartpole_artefacts.npy'):
        saved_artefacts = npload(filename, allow_pickle=True)
        return (saved_artefacts[()]['Q'], saved_artefacts[()]['successful_episode_index'])    

if __name__ == '__main__':
    logger = getLogger()
    logger.setLevel(INFO)
    logger.addHandler(TqdmLoggingHandler())

    cartpole_agent = QCartPoleAgent(verbose=True)
    episodes_taken = cartpole_agent.run(mode='train')