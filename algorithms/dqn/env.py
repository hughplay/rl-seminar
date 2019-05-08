from etr.utils import is_jupyter
import gym
import numpy as np
import torch


class PlayRecord:

    def __init__(self, gamma=0.99, device=torch.device('cpu')):
        self.gamma = gamma
        self.reset()
        self.device = device
        self.vars = [
            'x', 'action', 'log_prob', 'value', 'y',
            'reward', 'discount_reward', 'survive']

    def reset(self):
        self.steps = 0
        self.score = 0
        self.list_x = []
        self.list_action = []
        self.list_log_prob = []
        self.list_value = []
        self.list_y = []
        self.list_reward = []
        self.list_discount_reward = []
        self.list_survive = []

    def step(self, x, action, log_prob, value, y, reward, survive):
        self.steps += 1
        self.score += reward
        self.list_x.append(x)
        self.list_action.append(action)
        self.list_log_prob.append(log_prob)
        self.list_value.append(value)
        self.list_y.append(y)
        self.list_reward.append(reward)
        self.list_survive.append(survive)

    def done(self):
        self._compute_discount_reward()

    def _compute_discount_reward(self):
        R = 0
        self.list_discount_reward = []
        for reward in self.list_reward[::-1]:
            R =  reward + self.gamma * R
            self.list_discount_reward.append(R)

    def __len__(self):
        return self.steps

    def _transform(self, tensor):
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze_(1)
        return tensor.float().to(self.device)

    def __getattr__(self, key):
        if key in self.vars:
            var = getattr(self, 'list_{}'.format(key))
            var = torch.cat(
                var) if torch.is_tensor(var[0]) else torch.tensor(var)
            return self._transform(var)
        else:
            raise AttributeError(
                '\'{}\' object has no attribute \'{}\''.format(
                    self.__class__.__name__, key))

class Game:

    def __init__(
            self, name='CartPole-v0', seed=2019, gamma=0.99,
            device=torch.device('cpu')):
        self.name = name
        self.seed = seed
        self.gamma = gamma
        self.device = device

        self.env = gym.make(name)
        self.env.seed(self.seed)
        self.jupyter = is_jupyter()

        self.record = PlayRecord(self.gamma, self.device)
        self.cache_score = []
        self.done = False

        self.state = None
        self.fig = None
        self.str_state = None
        self.text_state = None

        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L48
        self.min_play = 100
        self.average_reward_threshold = 195.

    def reset(self):
        self.done = False
        self.record.reset()
        self.state = self.env.reset()
        return self.state

    def step(self, action, log_prob=0., value=0.):
        state, reward, self.done, _ = self.env.step(action.item())
        survive = 1. if self.env.steps_beyond_done is None else 0
        self.record.step(
            self.state, action, log_prob, value, state, reward, survive)

        self.state = state
        if self.over:
            self.cache_score.append(self.record.score)
            self.record.done()

    def render(self):
        if self.jupyter:
            score = self.record.score
            state = self.record.list_y[-1]
            if self.record.steps == 0:
                self.fig = plt.figure
                self.str_state = (
                    'Reward: {:.0f}\nCart Position: {:5.2f}\n'
                    'Cart Velocity: {:5.2f}\nPole Angle: {:5.2f}\n'
                    'Pole Velocity At Tip: {:5.2f}')
                self.text_state =  self.fig.text(
                    0.15, 0.6, self.str_state.format(ep_reward, *state))
                demo = plt.imshow(self.env.render(model='rgb_array'))
                plt.axis('off')
            self.text_state.set_text(self.str_state.format())
        else:
            self.env.render()

    @property
    def over(self):
        max_step = 10000
        return self.record.steps >= max_step or self.done

    @property
    def solved(self):
        if self.plays >= self.min_play:
            if self.average_score >= self.average_reward_threshold:
                return True
        return False

    @property
    def plays(self):
        return len(self.cache_score)

    @property
    def latest_score(self):
        return self.cache_score[-1]

    @property
    def average_score(self):
        return np.mean(self.cache_score[-self.min_play:])

    @property
    def state_tensor(self):
        return torch.from_numpy(self.state).float().to(self.device)
