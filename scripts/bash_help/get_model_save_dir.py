import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config_utils

config_path = sys.argv[1]
config = config_utils.read_config_from_user_file(config_path)

model_dir = os.path.dirname(config["model_savepath"])
print(model_dir)
