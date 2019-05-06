import argparse

from model import Model

class Trainer:

    def __init__(self, config):
        self.model = Model(**config['model'])

    def start_train(self):
        pass


if __name == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', default='config.yaml', help='Path of `config.yaml`.')

    args = parser.parse_args()
    config = get_config(args.config)

    trainer = Trainer(config)
    trainer.start_train()
