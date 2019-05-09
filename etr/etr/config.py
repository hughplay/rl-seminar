from datetime import datetime
import os
from pathlib import Path
from subprocess import check_output

from ruamel.yaml import YAML


class Config:

    yaml = None
    config = None

    def __init__(self, path, compute='auto'):
        self.path = Path(path)
        self.compute = compute

        self._init_yaml()
        self._read_yaml()

    def __contains__(self, key):
        return key in self.config

    def _init_yaml(self):
        self.yaml = YAML()

    def _read_yaml(self):
        self.config = self.yaml.load(self.path)
        if self.compute is True or (
                self.compute == 'auto' and not self.config['computed']):
            self.auto_compute()
            self.update()

    def update(self):
        self.save(self.path)

    def save(self, path):
        path = Path(path)
        self.yaml.dump(self.config, path)

    def init(self, name):
        self.config['computed'] = False
        self.config['name'] = name
        self.config['date'] = datetime.strftime(datetime.now(), '%Y-%m-%d')

    def get_id(self):
        strs = []
        for key in self.config['id_format']:
            strs.append(self.config[key])
        res = '-'.join(strs)
        if len(res) == 0:
            return hash(self.config)
        return res

    def get_group(self):
        """ Group experiment. """
        try:
            git_root = check_output(
               'git rev-parse --show-toplevel', shell=True).decode(
                   'utf-8').strip()
            rel_dir = os.path.relpath('.', git_root)
            group = os.path.dirname(rel_dir)
        except Exception:
            print('Warning: Not a git repository.')
            group = ''
        return group

    def auto_compute(self):
        self.config['id'] = self.get_id()

        if 'root' not in self.config:
            self.config['root'] = './train_log'
        self.config['group'] = self.get_group()
        self.config['exp_root'] = os.path.join(
            self.config['root'], self.config['group'], self.config['id'])
        self.config['ckpt_root'] = os.path.join(
            self.config['exp_root'], 'model')
        self.config['latest_model'] = os.path.join(
            self.config['ckpt_root'], 'latest.pth')
        self.config['log_root'] = os.path.join(self.config['exp_root'], 'log')
        self.config['tb_root'] = os.path.join(self.config['exp_root'], 'tb')

        self.config['computed'] = True

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise AttributeError(
                '\'{}\' object has no attribute \'{}\''.format(
                    self.__class__.__name__, key))


def get_config(path='config.yaml'):
    config_file = Path(path)
    if config_file.is_file():
        config = Config(config_file)
        return config
    else:
        raise FileNotFoundError(
            'You must run etr in a ETR project with `config.yaml`')
