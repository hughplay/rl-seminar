import argparse
from etr import get_config, get_logger
from pathlib import Path
import torch
from torch import optim

from model import Agent
from env import Game
from train import get_game


class Tester:

    def __init__(self, config):
        self.config = config
        self.agent = Agent(**self.config.agent)
        self.logger = get_logger(__name__)

        self.init_device()
        self.init_env()
        self.init_agent()

    def init_device(self):

        if self.config.cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info('Using gpu.')
        else:
            self.device = torch.device('cpu')
            self.logger.info('Using cpu.')

    def init_env(self):
        self.game = get_game(self.config, self.device)

    def init_agent(self):
        self.agent = Agent(**self.config.agent).to(self.device)
        self._load()

    def logging(self):
        self.logger.info(
            '{: >4}, score: {:6.2f} average score: {:6.2f}'.format(
                self.game.plays, self.game.latest_score,
                self.game.average_score))

    def _load(self):
        ckpt_path = Path(self.config.latest_model).expanduser()
        if ckpt_path.exists():
            self.logger.info(
                'Loading checkpoint from `{}`'.format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            self.agent.load_state_dict(ckpt['agent_state_dict'])
            self.logger.info('Loading complete.')
        else:
            raise FileNotFoundError('Checkpoint not found.')

    def start(self):
        try:
            self.agent.eval()

            self.logger.info('Start testing.')
            for i in range(self.config.n_test):
                self.game.reset()
                while not self.game.over:
                    action, log_prob, value = self.agent(
                        self.game.state_tensor)
                    self.game.step(action, log_prob, value)
                    if self.config.render_test:
                        self.game.render()
                self.logging()
            self.logger.info('Testing complete.')
        except KeyboardInterrupt:
            self.logger.warning('Keyboard Interrupt.')
        finally:
            self.logger.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', default='config.yaml',
        nargs='?', help='Path of `config.yaml`.')
    parser.add_argument(
        '--no-render', action='store_true', help='No rendering.')
    parser.add_argument(
        '-n', '--num', type=int, help='Number of testing.')

    args = parser.parse_args()
    config = get_config(args.config)
    if args.no_render:
        config.config['render_test'] = False
    if args.num:
        config.config['n_test'] = args.num

    tester = Tester(config)
    tester.start()
