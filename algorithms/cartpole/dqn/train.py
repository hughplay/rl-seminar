import argparse
from etr import get_config, get_logger
from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

from model import Agent
from env import Game


def get_optimizer(params, opt_config):
    optimizer = getattr(optim, opt_config['type'])
    return optimizer(params, **opt_config['args'])


def get_game(config, device):
    game = Game(
        config.task, config.seed, config.gamma,
        config.eval_episode, config.average_reward_threshold, device)
    return game


class Trainer:

    def __init__(self, config):
        self.config = config
        self.agent = Agent(**self.config.agent)
        self.logger = get_logger(__name__)

        self.gamma = float(self.config.gamma)
        self.epsilon = float(self.config.epsilon)

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

        num_action = self.config.num_action
        distribution = [
            self.epsilon / num_action] * num_action + [1 - self.epsilon]
        self.epsilon_sampler = Categorical(torch.tensor(distribution).float())

    def init_agent(self):
        torch.manual_seed(self.config.seed)
        self.agent = Agent(**self.config.agent).to(self.device)
        self.optimizer = get_optimizer(
            self.agent.parameters(), self.config.optimizer)
        self.loss = nn.MSELoss()

    def logging(self, force=False):
        if force or self.game.episode % self.config.log_interval == 0:
            self.logger.info(
                '{: >4}, score: {:6.2f} average score: {:6.2f} solved: {}'.format(
                    self.game.episode , self.game.latest_score,
                    self.game.average_score, self.game.solved))

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

    def epsilon_greedy(self, values):
        choice = self.epsilon_sampler.sample()
        if choice < self.config.num_action:
            action_idx = choice
        else:
            action_idx = torch.argmax(values)
        action = torch.zeros(self.config.num_action)
        action[action_idx] = 1.
        return action

    def get_data(self, state):
        num_action = self.config.num_action
        num_state = 1 if len(state.shape) == 1 else state.shape[0]
        if len(state.shape) == 1:
            state = state.unsqueeze_(0)
        state = torch.cat([state] * num_action)
        action = torch.cat(
            [torch.eye(num_action)] * num_state).to(self.device)
        return state, action

    def update_agent(self):
        self.optimizer.zero_grad()

        s_before = self.game.record.s_before
        action = self.game.record.action
        s_after = self.game.record.s_after
        reward = self.game.record.reward
        survive = self.game.record.survive
        steps = s_before.shape[0]

        value = self.agent(s_before, action)

        s_after, actions = self.get_data(s_after)
        values = self.agent(s_after, actions)
        values = torch.cat(values.split(steps), dim=1)
        max_values = values.max(dim=1, keepdim=True)
        y = reward + survive * self.gamma * max_values[0]

        output = self.loss(y, value)

        output.backward()
        self.optimizer.step()

    def start(self):
        try:
            self.agent.train()

            self.logger.info('Start training.')
            while (
                    self.game.episode < self.config.min_episode or
                    not self.game.solved):
                self.game.reset()
                while not self.game.over:
                    states, actions = self.get_data(self.game.state_tensor)
                    values = self.agent(states, actions)
                    action = self.epsilon_greedy(values)
                    self.game.step(action)
                    if self.config.render_train:
                        self.game.render()
                self.update_agent()
                self.logging()
            if self.game.solved:
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
