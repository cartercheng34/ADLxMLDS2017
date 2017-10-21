import os
import pandas as pd
import numpy as np
from random import shuffle

def read_data(path_d , path_l , path_m):
    file_path = os.path.join(path_d , 'train.ark')
    f = open(file_path , 'r')
    l = f.readline().split()
    cols = list(range(len(l)-1))
    cols.insert(0 , 'ID')    
    data = pd.read_csv(file_path , sep = ' ' , header = None , names = cols)
    
    label_path = os.path.join(path_l , 'train.lab')    
    cols3 = ['ID' , 'label_48']
    labels = pd.read_table(label_path , sep = ',' , header = None , names = cols3)
    
    train_48 = data.join(labels.set_index('ID') , on = 'ID')    
    
    map_path = os.path.join(path_m , '48_39.map')
    cols4 = ['label_48' , 'label_39']
    label_48_39 = pd.read_csv(map_path , sep = '\t' , header = None , names = cols4)    
    train_48 = train_48.join(label_48_39.set_index('label_48') , on = 'label_48')
    
    map_path = os.path.join(path_m , '48phone_char.map')
    cols2 = ['label_48' , 'index' , 'char']
    label_index_char = pd.read_csv(map_path , sep = '\t' , header = None , names = cols2)
    concatenate = train_48.join(label_index_char.set_index('label_48') , on = 'label_48')
        
    return concatenate
"""
def read_label(path)
    file_path = os.path.join(path , 'train.lab')    
    cols = ['ID' , 'label']
    labels = pd.read_table(file_path , sep = ',' , header = None , names = cols)
    return labels

def map_data(data , labels):
    
    train_48 = data.join(labels.set_index('ID') , on = 'ID')
    print(train_48)
    
    
    path = 'data/'
    file_path = os.path.join(path , '48_39.map')
    cols = ['label' , 'label_49']
    label_48_39 = pd.read_csv(file_path , sep = '\t' , header = None , names = cols)
    print(label_48_39)
    train_48 = data.join(label_48_39.set_index('label') , on = 'label')
    
    file_path = os.path.join(path , '48phone_char.map')
    cols2 = ['label' , 'index' , 'char']
    label_index_char = pd.read_csv(file_path , sep = '\t' , header = None , names = cols2)
    train_48 = data.join(label_index_char.set_index('label') , on = 'label')
    
    return train_48
"""

def gen_batch(data , params):
    batch_size = params['batch_size']
    feature_num = params['feature_num']

def gen_sequence(data , params):
    num_steps = params['num_steps']
    feature_num = params['feature_num']
    
    sequence = []
    labels = []
    j = 0 # record last index    
    
    for i in range(data.shape[0]):
        cur_speaker = data[i][0].split('_')[0]
        if i+1 == data.shape[0]:
            next_speaker == '' #???                        
        else:
            next_speaker = data[i+1][0].split('_')[0]
        
        if cur_speaker != next_speaker or i+1 == data.shape[0]:
            
            if (i-j+1) % num_steps != 0:
                diff = num_steps - ((i-j+1) % num_steps)
            else:
                diff = 0

            #print(cur_speaker)
            tmp_data = data[j:i+1 , 1:feature_num+1]
            #print(tmp_data.shape)            
            ll = np.zeros(feature_num)            
            for t in range(diff):
                tmp_data = np.append(tmp_data , ll)
            #print('diff=' , diff)
            if diff == 0:
                tmp_data = tmp_data.reshape(-1,1)
            #print(len(tmp_data))
            tmp_data = np.asarray(tmp_data)
            tmp_data = tmp_data.reshape(int(len(tmp_data)/num_steps/feature_num) , num_steps , feature_num)
            
            

            tmp_label = data[j:i+1 , feature_num+3]            
            kk = [37] #sil
            for s in range(diff):
                tmp_label = np.append(tmp_label , kk)            
            tmp_label = np.asarray(tmp_label)
            
            tmp_label = tmp_label.reshape(int(len(tmp_label)/num_steps) , num_steps)
            
            j = i+1
            for x in range(tmp_data.shape[0]):
                sequence.append(tmp_data[x])
            for y in range(tmp_label.shape[0]):
                labels.append(tmp_label[y])
    
    return sequence , labels

def cross_validation(sequence , params):
    training_ratio = params['training_ratio']    
    training_len = int(len(sequence) * training_ratio)
    return sequence[:training_len] ,sequence[training_len:] 


def gen_batch(sequence , labels ,params):
    num_steps = params['num_steps']
    batch_size = params['batch_size']
    feature_num = params['feature_num']
    
    shuffle_index = list(range(len(sequence)))
    
    shuffle(shuffle_index)
    sequence = np.array(sequence)
    labels = np.array(labels)
    sequence = sequence[shuffle_index]
    labels = labels[shuffle_index]
    print('ori=' , len(sequence))
    diff = batch_size - len(sequence) % batch_size
    print('diff=' , diff)
    tmp_seq = sequence
    tmp_label = labels
    for i in range(diff):
        tmp_seq = np.append(tmp_seq , sequence[i])
        tmp_label = np.append(tmp_label , labels[i])
    print('len=' , len(tmp_seq))       
    batch_data = np.array(tmp_seq).reshape(int(len(tmp_seq)/batch_size/num_steps/feature_num) , batch_size , num_steps , feature_num)
    batch_label = np.array(tmp_label).reshape(int(len(tmp_label)/batch_size/num_steps) , batch_size , num_steps)
    #total_batch = np.reshape(sequence, [len(sequence)/batch_size, batch_size, num_steps, numOfFeatures])
    
    return batch_data , batch_label