import json
from easydict import EasyDict
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    path = os.path.dirname(__file__)
    config.test_summary_dir = os.path.join(
        path, "../logs", config['exp_name'], "test/")
        # "../logs", config['exp_name'], "test/")
    config.train_summary_dir = os.path.join(
        path, "../logs", config['exp_name'], "train/")
        #"../logs", config['exp_name'], "train/")
    config.checkpoint_dir = os.path.join(
        path, "../weights", config['exp_name'] + "/")

        #"../weights", config['exp_name'] + "/")
    return config
