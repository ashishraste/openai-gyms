from sklearn.model_selection import ParameterGrid
from numpy import mean, argmax, argmin, zeros
from multiprocessing import Pool, cpu_count
from cartpole_agent import QCartPoleAgent
from tqdm_logger import TqdmLoggingHandler
from logging import getLogger, INFO, info

candidate_params = {
    'min_alpha': [0.1, 0.2],
    'min_epsilon': [0.1, 0.2],
    'state_buckets': [(1, 1, 6, 3), (1, 1, 3, 6), (1, 1, 6, 12), (1, 1, 12, 6)],
    'episode_factor': [25, 30],
    'gamma': [1.0, 0.99]
}

search_grid = list(ParameterGrid(candidate_params))
final_scores = zeros(len(search_grid))

def evaluate_cartpole_agent(args):
    index, params = args
    info('Evaluating parameter set: {}'.format(params))
    params = {**params}

    scores = []
    num_runs_per_sample_set = 10
    for i in range(num_runs_per_sample_set):
        agent = QCartPoleAgent(**params)
        score = agent.run(disable_progress_bar=True)
        scores.append(score)

    episodes_taken = mean(scores)
    info('Parameter set {} finished at {} episodes'.format(params, episodes_taken))
    return episodes_taken

def run_grid_search():    
    info('Starting grid-search over {} parameter sets'.format(len(search_grid)))
    pool = Pool(processes=cpu_count())
    final_scores = pool.map(evaluate_cartpole_agent, list(enumerate(search_grid)))

    info('Best parameter set was {} which took an average of {} episodes to solve'.format(search_grid[argmin(final_scores)], min(final_scores)))
    info('Worst parameter set was {} which took an average of {} episodes to solve'.format(search_grid[argmax(final_scores)], max(final_scores)))

if __name__ == '__main__':
    logger = getLogger()
    logger.setLevel(INFO)
    logger.addHandler(TqdmLoggingHandler())

    run_grid_search()