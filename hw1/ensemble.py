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
    state_size = 512,
    num_layers = 2,
    num_classes = 48,
    batch_size = 16,
    num_epochs = 35,
    num_steps = 20,
    learning_rate = 0.0001,
    training_ratio = 0.9,
    feature_num = 69,
    cnn_filter_num = [32 , 32],
    cnn_filter_size = 3,
    cnn_pool_size = [2 , 2],
    cnn_fc_layer_size = [512 , 256]
    )
params2 = dict(
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
    cnn_filter_num = [32 , 32 , 32],
    cnn_filter_size = 3,
    cnn_pool_size = [2 , 2],
    cnn_fc_layer_size = [512 , 256 , 128]
    )

params3 = dict(
    input_size = 28,
    state_size = 256,
    num_layers = 2,
    num_classes = 48,
    batch_size = 16,
    num_epochs = 40,
    num_steps = 20,
    learning_rate = 0.0001,
    training_ratio = 0.9,
    feature_num = 69,
    cnn_filter_num = [32 , 32],
    cnn_filter_size = 5,
    cnn_pool_size = [2 , 2],
    cnn_fc_layer_size = [512 , 256]
    )


"""
params2 = dict(
    state_size = 512,
    num_layers = 2,
    num_classes = 48,
    batch_size = 16,
    num_epochs = 35,
    num_steps = 20,
    learning_rate = 0.0001,
    training_ratio = 0.9,
    feature_num = 69,
    cnn_filter_num = [50 , 50],
    cnn_filter_size = 3,
    cnn_pool_size = [2 , 2],
    cnn_fc_layer_size = [512 , 256]
    )
"""
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
        

def predict(test_seq , params , model_path , max_len , max_indices , map48_char , ids , ckpt_path , graph):
    batch_size = params['batch_size']
    test_batch , batch_indices , diff = util.gen_test_batch(test_seq , params , max_len , max_indices )
    num_classes = params['num_classes']  

    pred_output = []
    with tf.Session() as sess:
        model_saver = tf.train.import_meta_graph(model_path)
        model_saver.restore(sess , tf.train.latest_checkpoint(ckpt_path))     
                
        #graph = tf.get_default_graph()
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        sequence_length = graph.get_tensor_by_name('sequence_length:0')        
        x = graph.get_tensor_by_name('input_placeholder:0')
        predictions = tf.get_collection('predictions')[0]   
        predictions_one_hot = tf.get_collection('predictions_one_hot')[0]        
        
        for i in range(len(test_batch)):
            print('test_batch:' , (test_batch[i].shape))
            #print('x:' , test_batch[i])
            feed_dict = {x:test_batch[i] , keep_prob:1 , sequence_length:batch_indices[i]}
            predict_ = sess.run(predictions_one_hot , feed_dict = feed_dict)
            print('predict =' , predict_)
            print('shape:' , np.array(predict_).shape)
            predict_ = np.array(predict_)
            pred_output.append(predict_) 
         # len_test_batch x batch_size x max_len
        print('total_predict:' , np.array(pred_output).shape)        
        #print('ids:' , len(ids)) #592
        print('indices:' , max_indices.shape)
        pred_output = np.array(pred_output) #592x777x48
        return np.reshape(pred_output , [-1 , batch_size , max_len , num_classes])
        
        
                    


model_path = ['model/cnn-fc-512-256.ckpt.meta' , 'model3/cnn_with3layer.ckpt.meta' , 'model2/cnn-32-5-256.ckpt.meta']
ckpt_path = ['model' , 'model3' , 'model2']
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

g1 = tf.Graph()
g2 = tf.Graph()
g3 = tf.Graph()
with g1.as_default():
    model1_output = predict(test_seq , params , model_path[0] , max_len , max_indices , map48_char , ids , ckpt_path[0] , g1)
    print('model1_output' , model1_output.shape)
with g2.as_default():
    model2_output = predict(test_seq , params2 , model_path[1] , max_len , max_indices , map48_char , ids , ckpt_path[1] , g2)
    print('model2_output' , model2_output.shape)
with g3.as_default():
    model3_output = predict(test_seq , params3 , model_path[2] , max_len , max_indices , map48_char , ids , ckpt_path[2] , g3)

tmp = np.multiply(model1_output , model2_output)
tmp = np.multiply(tmp , model3_output)
pred_output = np.argmax(tmp , axis = 3)

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
