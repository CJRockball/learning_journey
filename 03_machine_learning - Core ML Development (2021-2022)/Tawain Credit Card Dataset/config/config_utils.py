import os
import pathlib

import yaml

# folder to load config file
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "config/"
# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_DIR, config_name)) as file:
        config = yaml.safe_load(file)
    return config
