import argparse
from etr import get_config, get_logger
from pathlib import Path
import torch
from torch import optim

from model import Agent
from env import Game


def get_optimizer(params, opt_config):
    optimizer = getattr(optim, opt_config['type'])
    return optimizer(params, **opt_config['args'])


def get_game(config, device):
    game = Game(config.task, config.seed, config.gamma, device)
    return game


class Trainer:

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
        torch.manual_seed(self.config.seed)
        self.agent = Agent(**self.config.agent).to(self.device)
        self.optimizer = get_optimizer(
            self.agent.parameters(), self.config.optimizer)

    def logging(self, force=False):
        if force or self.game.plays % self.config.log_interval == 0:
            self.logger.info(
                '{: >4}, score: {:6.2f} average score: {:6.2f}'.format(
                    self.game.plays, self.game.latest_score,
                    self.game.average_score))

    def load(self):
        ckpt_path = Path(self.config.latest_model).expanduser()
        if ckpt_path.exists():
            self.logger.info(
                'Loading checkpoint from `{}`'.format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            self.game.cache_score = ckpt['cache_score']
            self.agent.load_state_dict(ckpt['agent_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.logger.info('Loading complete.')
            self.logging(True)

    def save(self):
        Path(self.config.ckpt_root).expanduser().mkdir(
            parents=True, exist_ok=True)
        ckpt_path = Path(self.config.latest_model).expanduser()
        torch.save({
            'cache_score': self.game.cache_score,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, ckpt_path)

        self.logger.info(
            'Checkpoint saved in `{}`.'.format(ckpt_path))
        self.logging(True)

    def update_agent(self):
        pass

    def start(self):
        try:
            self.agent.train()

            self.logger.info('Start training.')
            while not self.game.solved:
                self.game.reset()
                while not self.game.over:
                    action, log_prob, value = self.agent(
                        self.game.state_tensor)
                    self.game.step(action, log_prob, value)
                    if self.config.render_train:
                        self.game.render()
                self.update_agent()
                self.logging()
            self.logger.info('{} has been solved!'.format(self.game.name))
            self.logger.info('Training complete.')
        except KeyboardInterrupt:
            self.logger.warning('Keyboard Interrupt.')
        finally:
            self.save()
            self.logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', default='config.yaml',
        nargs='?', help='Path of `config.yaml`.')
    parser.add_argument(
        '--new', action='store_true', help='Whether train from scratch')

    args = parser.parse_args()
    config = get_config(args.config)

    trainer = Trainer(config)
    if not args.new:
        trainer.load()
    trainer.start()
