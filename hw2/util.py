import os
import json


def parse(path):
    #file_path = params['training_data_path']
    with open(path , 'r') as f:
        captions = json.load(f)
    return captions

