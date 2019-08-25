import argparse
from easydict import EasyDict

from agents import *
from utils.config import get_config_from_json


def main(config):
    """
    Args:
        config (Dict or EasyDict)
    """
    if isinstance(config, EasyDict) is False:
        config = EasyDict(config)
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(**config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    config = get_config_from_json(args.config)

    main(config)
