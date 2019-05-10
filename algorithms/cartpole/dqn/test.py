import argparse
from etr import get_config, get_logger
import numpy as np
from pathlib import Path
import torch
from torch import optim

from model import Agent
from env import Game
from train import Trainer


class Tester(Trainer):

    def init_agent(self):
        self.agent = Agent(**self.config.agent).to(self.device)
        self.load()

    def load(self):
        ckpt_path = Path(self.config.latest_model).expanduser()
        if ckpt_path.exists():
            self.logger.info(
                'Loading checkpoint from `{}`'.format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            self.agent.load_state_dict(ckpt['agent_state_dict'])
            self.logger.info('Loading complete.')
        else:
            raise FileNotFoundError('Checkpoint not found.')

    def logging(self):
        self.logger.info(
            '{: >3}, score: {:6.2f} average score: {:6.2f}'.format(
                self.game.episode, self.game.latest_score,
                self.game.average_score))

    def greedy(self, values):
        action_idx = torch.argmax(values)
        action = torch.zeros(self.config.num_action)
        action[action_idx] = 1.
        return action

    def start(self):
        try:
            self.agent.eval()

            self.logger.info('Start testing.')
            for i in range(self.config.n_test):
                self.game.reset()
                while not self.game.over:
                    states, actions = self.get_data(self.game.state_tensor)
                    values = self.agent(states, actions)
                    action = self.epsilon_greedy(values)
                    self.game.step(action)
                    if self.config.render_test:
                        self.game.render()
                self.logging()
            self.logger.info('Testing complete.')
        except KeyboardInterrupt:
            self.logger.warning('Keyboard Interrupt.')
        finally:
            self.logger.info('Total survived episodes: {}/{}'.format(
                np.sum(np.array(self.game.cache_score) >= 200),
                self.config.n_test))
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
