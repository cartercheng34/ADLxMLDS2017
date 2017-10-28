import util
import model_cnn
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import sys

path = os.path.join(sys.argv[1] , 'fbank/')
lab_path = 'data/'
map_char_path = os.path.join(sys.argv[1] , '48phone_char.map')
map_39_path = os.path.join(sys.argv[1] , 'phones/48_39.map')
os.environ['CUDA_VISIBLE_DEVICES']='0'

params = dict(
    input_size = 28,
    state_size = 512,
    num_layers = 2,
    num_classes = 48,
    batch_size = 16,
    num_epochs = 40,
    num_steps = 20,
    learning_rate = 0.0001,
    training_ratio = 0.9,
    feature_num = 69,
    cnn_filter_num = [32 , 32],
    cnn_filter_size = 3,
    cnn_pool_size = [2 , 2],
    cnn_fc_layer_size = [250 , 100]
    )

def phone48_39(seq , map):
    merge = np.full(len(seq) , '0' , dtype='U5')
    for i , phone in enumerate(map[: , 0]):
        merge[seq == phone] = map[i , 1]
    return merge

def index_phone(seq , map):
    merge = np.full(len(seq) , '0' , dtype='U5')
    for i , char in enumerate(map[: , 1]):
        merge[seq == char] = map[i , 0]    
    return merge

def phone_char(seq , map):
    merge = np.full(len(seq) , '0' , dtype='U1')
    for i , char in enumerate(map[: , 0]):
        merge[seq == char] = map[i , 2]
    merge = ''.join(merge)
    return merge

def trim(seq , max_indices ,counter):
    seq = seq[:int(max_indices[0 , counter])]    
    trim = ''.join([match[0] + match[1] for match in re.findall(r'(\w)(\1+)', seq)])    
    trim = ''.join([match[0] + match[1] for match in re.findall(r'(\w)(\1{2,})', trim)])    
    #trim = ''.join([match[0] + match[1] for match in re.findall(r'(\w)(\1{3,})', trim)])
    
    trim = re.sub(r'(\w)\1+', r'\1', trim)
    trim = re.sub(r'^L', '', trim) # remove leading sil
    #trim = re.sub(r'a$', '', trim) # remove tailing a
    trim = re.sub(r'L$', '', trim) # remove tailing sil
    return trim
        

def predict(test_seq , params , model_path , max_len , max_indices , map48_char , ids):
    batch_size = params['batch_size']
    test_batch , batch_indices , diff = util.gen_test_batch(test_seq , params , max_len , max_indices )    

    pred_output = []
    with tf.Session() as sess:
        model_saver = tf.train.import_meta_graph('model/cnn-fc-512-256.ckpt.meta')
        model_saver.restore(sess , tf.train.latest_checkpoint('model/'))
        #x = tf.get_collection('x')
        #keep_prob = tf.get_collection('keep_prob')
        #predictions = tf.get_collection('predictions')
        graph = tf.get_default_graph()
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        sequence_length = graph.get_tensor_by_name('sequence_length:0')
        print('tensor:' , type(keep_prob))
        x = graph.get_tensor_by_name('input_placeholder:0')
        predictions = tf.get_collection('predictions')[0]   
        predictions_one_hot = tf.get_collection('predictions_one_hot')[0]
        print(type(keep_prob))
        print(type(x))
        
        for i in range(len(test_batch)):
            print('test_batch:' , (test_batch[i].shape))
            #print('x:' , test_batch[i])
            feed_dict = {x:test_batch[i] , keep_prob:1 , sequence_length:batch_indices[i]}
            predict_ = sess.run(predictions , feed_dict = feed_dict)
            print('predict =' , predict_)
            print('shape:' , np.array(predict_).shape)
            predict_ = np.array(predict_)
            pred_output.append(predict_) 
         # len_test_batch x batch_size x max_len
        print('total_predict:' , np.array(pred_output).shape)
        #pred_output = pred_output[:len(pred_output)-diff] # discard the last diff batch
        #print('ids:' , len(ids)) #592
        print('indices:' , max_indices.shape)
        pred_output = np.array(pred_output) #19x32
        counter = 0
        with open(sys.argv[2] , 'w') as f:
            print('id,phone_sequence' , file = f)
            for i in range(pred_output.shape[0]):
                for j in range(pred_output.shape[1]):
                    if counter < len(ids):                        
                        sentence = index_phone(pred_output[i][j] , map48_char)                    
                        if counter == 1:
                            print(sentence[0])
                        sentence = phone48_39(sentence , map48_39)
                        if counter == 1:
                            print(sentence)
                        sentence = phone_char(sentence , map48_char)
                        if counter == 1:
                            print(sentence)
                        trimmed = trim(sentence , max_indices , counter)
                        if counter == 1:
                            print(trimmed)
                        print('%s,%s' % (ids[counter],trimmed) , file = f)                      
                        counter += 1
                    


model_path = 'model/'
test_path = os.path.join(sys.argv[1] , 'fbank/')
"""
data , map48_39 , map48_char = util.read_data(path , lab_path , map_path)
map48_39 = np.array(map48_39) 
map48_char = np.array(map48_char)
"""
test_data = util.read_test(test_path)
map48_char = np.array(pd.read_csv(map_char_path, header=None, sep='\t'))
map48_39 = np.array(pd.read_csv(map_39_path, header=None, sep='\t'))

test_seq , max_len , max_indices , ids = util.gen_sequence(test_data.values , params , False)
predict(test_seq , params , model_path , max_len , max_indices , map48_char , ids)
