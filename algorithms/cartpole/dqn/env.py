from etr.utils import is_jupyter
import gym
import numpy as np
import torch


class PlayRecord:

    def __init__(self, gamma=0.99, device=torch.device('cpu')):
        self.gamma = gamma
        self.reset()
        self.device = device

    def reset(self):
        self.steps = 0
        self.score = 0
        self.list_s_before = []
        self.list_action = []
        self.list_reward = []
        self.list_s_after = []
        self.list_survive = []

    def step(self, s_before, action, reward, s_after, survive):
        self.steps += 1
        self.score += reward
        self.list_s_before.append(s_before)
        self.list_action.append(action.unsqueeze_(0))
        self.list_reward.append(reward)
        self.list_s_after.append(s_after)
        self.list_survive.append(survive)

    def __len__(self):
        return self.steps

    def _transform(self, tensor):
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze_(1)
        return tensor.float().to(self.device)

    def __getattr__(self, key):
        list_key = 'list_{}'.format(key)
        if hasattr(self, list_key):
            var = getattr(self, list_key)
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
            eval_episode=100, average_reward_threshold=195.,
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

        self.eval_episode = eval_episode
        self.average_reward_threshold = average_reward_threshold

    def reset(self):
        self.done = False
        self.record.reset()
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        action_idx = torch.argmax(action).item()
        state, reward, self.done, _ = self.env.step(action_idx)
        survive = 1. if self.env.steps_beyond_done is None else 0
        self.record.step(self.state, action, reward, state, survive)

        self.state = state
        if self.over:
            self.cache_score.append(self.record.score)

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
        if self.episode >= self.eval_episode:
            if self.average_score >= self.average_reward_threshold:
                return True
        return False

    @property
    def episode(self):
        return len(self.cache_score)

    @property
    def latest_score(self):
        return self.cache_score[-1]

    @property
    def average_score(self):
        return np.mean(self.cache_score[-self.eval_episode:])

    @property
    def state_tensor(self):
        return torch.from_numpy(self.state).float().to(self.device)
