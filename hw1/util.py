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
        
    return concatenate , label_48_39 , label_index_char

def read_test(path_d):
    file_path = os.path.join(path_d , 'test.ark')
    f = open(file_path , 'r')
    l = f.readline().split()
    cols = list(range(len(l)-1))
    cols.insert(0 , 'ID')    
    data = pd.read_csv(file_path , sep = ' ' , header = None , names = cols)

    return data
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

def gen_sequence(data , params , contain_label = True):
    num_steps = params['num_steps']
    feature_num = params['feature_num']
    overlap = params['overlap']
    sequence = []
    labels = []
    max_len = 0    
    j = 0 # record last index    
    if contain_label == True:
        for i in range(data.shape[0]):
            cur_sen = data[i][0].split('_')[0]+ '_' + data[i][0].split('_')[1]
            if i+1 == data.shape[0]:
                next_sen == '' #???                        
            else:
                next_sen = data[i+1][0].split('_')[0]+ '_' + data[i+1][0].split('_')[1]                
            
            if cur_sen != next_sen or i+1 == data.shape[0]:
                #print(cur_speaker)
                tmp_data = data[j:i+1 , 1:feature_num+1]
                tmp_label = data[j:i+1 , feature_num+3]#48 labels
                #print(tmp_data.shape)      
                    
                sequence.append(tmp_data)
                labels.append(tmp_label)
                j = i+1
        
        for i in range(len(sequence)):
            if len(sequence[i]) > max_len:
                max_len = len(sequence[i])
        #append 0
        
        output_seq = []
        output_label = []              
        max_indices = np.zeros((1,len(sequence)))

        for i in range(len(sequence)):         
            max_indices[0 , i] = max_len - len(sequence[i])   
            for t in range(max_len - len(sequence[i])):                                                            
                sequence[i]  = np.concatenate((sequence[i] , np.zeros((1,feature_num))) , axis = 0)
                labels[i] = np.concatenate((labels[i] , np.zeros(1)) , axis = 0)                                
            output_seq.append(sequence[i])
            output_label.append(labels[i])
        #print('output_seq:' , output_seq[0])
        print('shape:' , np.array(output_seq[0]).shape) #30303
        print('output_seq: ' , np.array(output_seq).shape)
        
        output_seq = np.reshape(np.array(output_seq), [-1 , max_len , feature_num])
        output_label = np.reshape(output_label, [-1 , max_len])
        max_indices = np.array(max_indices)
        return output_seq , output_label , max_len , max_indices
    else:
        for i in range(data.shape[0]):
            cur_sen = data[i][0].split('_')[0]+ '_' + data[i][0].split('_')[1]
            if i+1 == data.shape[0]:
                next_sen == '' #???                        
            else:
                next_sen = data[i+1][0].split('_')[0]+ '_' + data[i+1][0].split('_')[1]                
            
            if cur_sen != next_sen or i+1 == data.shape[0]:
                #print(cur_speaker)
                tmp_data = data[j:i+1 , 1:feature_num+1]
                
                #print(tmp_data.shape)                          
                sequence.append(tmp_data)                
                j = i+1
        
        for i in range(len(sequence)):
            if len(sequence[i]) > max_len:
                max_len = len(sequence[i])
        #append 0
        
        output_seq = []                      
        max_indices = np.zeros((1,len(sequence)))

        for i in range(len(sequence)):         
            max_indices[0 , i] = max_len - len(sequence[i])   
            for t in range(max_len - len(sequence[i])):                                                            
                sequence[i]  = np.concatenate((sequence[i] , np.zeros((1,feature_num))) , axis = 0)                                                
            output_seq.append(sequence[i])            
        #print('output_seq:' , output_seq[0])
        print('shape:' , np.array(output_seq[0]).shape) #30303
        print('output_seq: ' , np.array(output_seq).shape)
        
        output_seq = np.reshape(np.array(output_seq), [-1 , max_len , feature_num])        
        max_indices = np.array(max_indices)
        return output_seq , max_len , max_indices

def cross_validation(sequence , params):
    training_ratio = params['training_ratio']    
    training_len = int(len(sequence) * training_ratio)
    return sequence[:training_len] ,sequence[training_len:] 


def gen_batch(sequence , labels ,params , max_len , max_indices):
    num_steps = params['num_steps']
    batch_size = params['batch_size']
    feature_num = params['feature_num']
    
    shuffle_index = list(range(len(sequence)))
    print('shu:' , len(shuffle_index))
    shuffle(shuffle_index)
    sequence = np.array(sequence)
    labels = np.array(labels)
    sequence = sequence[shuffle_index]
    labels = labels[shuffle_index]
    max_indices = max_indices[0 , shuffle_index]
    print('ori=' , len(sequence))
    diff = batch_size - len(sequence) % batch_size
    print('diff=' , diff)
    tmp_seq = sequence
    tmp_label = labels
    tmp_indices = max_indices
    for i in range(diff):
        tmp_seq = np.append(tmp_seq , sequence[i])
        tmp_label = np.append(tmp_label , labels[i])
        tmp_indices = np.append(tmp_indices , max_indices[i])
    print('len=' , len(tmp_seq))       
    batch_data = np.reshape(tmp_seq , [-1 , batch_size , max_len , feature_num])
    batch_label = np.reshape(tmp_label , [-1 , batch_size , max_len])
    batch_indices = np.reshape(tmp_indices , [-1 , batch_size])
    #total_batch = np.reshape(sequence, [len(sequence)/batch_size, batch_size, num_steps, numOfFeatures])
    
    return batch_data , batch_label , batch_indices

def gen_test_batch(sequence , params , max_len , max_indices):
    num_steps = params['num_steps']
    batch_size = params['batch_size']
    feature_num = params['feature_num']
    
    diff = batch_size - len(sequence) % batch_size
   
    tmp_seq = sequence
    
    tmp_indices = max_indices
    for i in range(diff):
        tmp_seq = np.append(tmp_seq , sequence[i])        
        tmp_indices = np.append(tmp_indices , max_indices[i])
    print('len=' , len(tmp_seq))       
    batch_data = np.reshape(tmp_seq , [-1 , batch_size , max_len , feature_num])    
    batch_indices = np.reshape(tmp_indices , [-1 , batch_size])
    return test_batch