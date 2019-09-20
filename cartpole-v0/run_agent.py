from qcartpole_agent import QCartPoleAgent
from logging import getLogger, INFO, DEBUG, info, debug, error
from numpy import load as npload
from tqdm_logger import TqdmLoggingHandler
from argparse import ArgumentParser

def load_trained_artefacts(filename='cartpole_artefacts.npy'):
        saved_artefacts = npload(filename, allow_pickle=True)
        return (saved_artefacts[()]['Q'], saved_artefacts[()]['successful_episode_index'])

if __name__ == '__main__':
    logger = getLogger()
    logger.setLevel(INFO)
    logger.addHandler(TqdmLoggingHandler())

    parser = ArgumentParser(description='Parse agent arguments.')
    parser.add_argument('--mode', default='train', help='Run agent in training or testing mode. \
        Accepts either [train] or [test] as input argument.')
    argspace = parser.parse_args()
    mode = argspace.mode

    if mode == 'train':
        cartpole_agent = QCartPoleAgent(verbose=True)
        cartpole_agent.run(mode=mode, render=True)
        cartpole_agent.save_trained_artefacts()
    elif mode == 'test':
        # Run the agent with pre-trained artefacts.
        qtable, num_ep = load_trained_artefacts()
        cartpole_agent = QCartPoleAgent(verbose=True)
        cartpole_agent.run(mode=mode, successful_episode_index=num_ep, render=True, saved_q_table=qtable)
    else:
        error('Unknown argument \'{}\' given for run mode, quitting ...'.format(mode))
        parser.print_help()
        exit(-1)