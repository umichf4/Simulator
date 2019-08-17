# -*- coding: utf-8 -*-

import argparse
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
#import utils
import json
from train import train_simulator


parser = argparse.ArgumentParser()
parser.add_argument('--save_model_dir', default='models', type=str)
parser.add_argument('--restore_from', default=None, type=str)
parser.add_argument('--json_path', default='params.json', type=str)
args = parser.parse_args()

# Load parameters from json file
json_path = args.json_path
assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
params = json.load(open(json_path))

train_simulator(params, args)