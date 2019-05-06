# -*- coding: utf-8 -*-

__version__ = "0.1.0"

import os
from pathlib import Path
import shutil
import sys
from subprocess import call

import argparse

from .config import get_config

package_dir = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(package_dir, 'template')

default_config_file = 'config.yaml'


def create(args):
    name = args.name
    target_path = Path(name)
    if target_path.exists():
        raise FileExistsError('{} already exists.'.format(name))
    else:
        shutil.copytree(
            example_dir, target_path,
            ignore=shutil.ignore_patterns('__pycache__'))


def init(args):
    name = os.path.basename(os.path.realpath('.'))
    config = get_config(default_config_file)
    old_name = config['name']
    if config and not args.nogit:
        git_commit(name, 'Adventure starts from #{}.'.format(old_name), 'tada')
    config.init(name)
    config.update()
    print('Experiment has been initialized.')


def git_commit(name=None, comment='', mark='beers'):
    message = ':beers: #{}: {}'.format(name, comment)
    call('git add . && git commit -m "%s"' % message, shell=True)


def commit(args):
    config = get_config(default_config_file)
    name = config.exp['name']
    comment = args.message if args.message else 'Mark and have a rest.'
    mark = args.icon if args.icon else 'beers'
    git_commit(name, comment=comment, mark=mark)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_new = subparsers.add_parser('new')
    parser_new.set_defaults(func=create)
    parser_new.add_argument('name', help='experiment name.')

    parser_init = subparsers.add_parser('init')
    parser_init.set_defaults(func=init)
    parser_init.add_argument(
        '-m', '--message', default=":beers: start from here.",
        help='git commit message.')
    parser_init.add_argument(
        '--nogit', action='store_true', help='git commit message.')

    parser_commit = subparsers.add_parser('commit')
    parser_commit.set_defaults(func=commit)
    parser_commit.add_argument(
        '-m', '--message', default=None, help='git commit message.')
    parser_commit.add_argument(
        '-i', '--icon', default=None, help='gitmoji')

    args = parser.parse_args()
    args.func(args)
