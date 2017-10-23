import util
import rnn
import os
import numpy as np
import tensorflow as tf
path = 'data/mfcc/'
lab_path = 'data/'
map_path = 'data/'


params = dict(
    input_size = 28,
    hidden_size = 128,
    state_size = 100,
    num_layers = 2,
    num_classes = 48,
    batch_size = 32,
    num_epochs = 50,
    num_steps = 20,
    learning_rate = 0.0001,
    training_ratio = 0.9,
    overlap = 10,
    feature_num = 39)

def phone48_39(seq , map):
    merge = np.full(len(seq) , 0 , dtype='U5')
    for i , phone in enumerate(map[: , 0]):
        merge[seq == phone] = map[i][1]
    return merge

def phone_char(seq , map):
    merge = np.full(len(seq) , 0 , dtype='U1')
    for i , phone in enumerate(map[: , 0]):
        merge[seq == phone] = map[i][2]
    merge = ''.join(merge)
    return merge

def trim(seq):
    print()

def predict(test_seq , params , model_path):
    
    test_batch = util.gen_test_batch(test_seq , params)    

    pred_output = []
    with tf.Session() as sess:
        model_saver = tf.train.import_meta_graph('model/layer2-r.ckpt.meta')
        model_saver.restore(sess , tf.train.latest_checkpoint('model/'))
        #x = tf.get_collection('x')
        #keep_prob = tf.get_collection('keep_prob')
        #predictions = tf.get_collection('predictions')
        graph = tf.get_default_graph()
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        print('tensor:' , type(keep_prob))
        x = graph.get_tensor_by_name('input_placeholder:0')
        predictions = tf.get_collection('predictions')[0]   
        #predictions_one_hot = tf.get_collection('predictions_one_hot')
        print(type(keep_prob))
        print(type(x))
        
        for i in range(len(test_batch)):
            print('test_batch:' , (test_batch[i].shape))
            #print('x:' , test_batch[i])
            feed_dict = {x:test_batch[i] , keep_prob:1}
            predict_ = sess.run(predictions , feed_dict = feed_dict)
            print('predict =' , predict_)
            pred_output.append(predict_)
        print('total_predict:' , np.array(pred_output).shape)


model_path = 'model/'
test_path = 'data/mfcc'
"""
data , map48_39 , map48_char = util.read_data(path , lab_path , map_path)
map48_39 = np.array(map48_39) 
map48_char = np.array(map48_char)
"""
test_data = util.read_test(test_path)
test_seq = util.gen_sequence(test_data.values , params , False)

predict(test_seq , params , model_path)