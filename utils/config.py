import json
from easydict import EasyDict


def get_config_from_json(json_file):
    """Get the config from a json file
    Args:
        json_file (str): Path of json file
    Returns:
        config (EasyDict[str, any]): Config
    """

    with open(json_file, 'r') as f:
        try:
            config = json.load(f)
            config = EasyDict(config)
            return config
        except:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)
