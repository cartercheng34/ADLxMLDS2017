import os
import json


def parse(path):
    #file_path = params['training_data_path']
    with open(path , 'r') as f:
        captions = json.load(f)
    return captions

def parse_peer_id(path):
    file = open(path , 'r')
    #done = 0
    ids = []
    for line in file.readlines():
        tmp = line.split('\n')
        ids.append(tmp[0])
    
    file.close()
    return ids
        

