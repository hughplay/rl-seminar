from datetime import datetime
from pathlib import Path

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

    def auto_compute(self):
        self.config['computed'] = True
        pass

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
