
import util
import rnn
import os
import numpy as np
import tensorflow as tf

# Hyper Parameters
os.environ['CUDA_VISIBLE_DEVICES']='1'

input_size = 28
hidden_size = 128
num_layers = 3
num_classes = 48
batch_size = 16
num_epochs = 40
num_steps = 20
learning_rate = 0.0001
training_ratio = 0.9
feature_num = 39
state_size = 512
overlap = 10

params = dict(
    input_size = 28,
    hidden_size = 128,
    state_size = 512,
    num_layers = 3,
    num_classes = 48,
    batch_size = 16,
    num_epochs = 35,
    num_steps = 20,
    learning_rate = 0.0001,
    training_ratio = 0.9,
    overlap = 10,
    feature_num = 39)

def train(params , sequence , labels , graph , model_path , max_len , max_indices):
    
    #batch_data , batch_label , batch_indices = util.gen_batch(sequence , labels , params , max_len , max_indices)
    
    #train_batch , valid_batch = util.cross_validation(batch_data , params)
    #train_label , valid_label = util.cross_validation(batch_label , params)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        
        model_saver = tf.train.Saver()

        for i in range(num_epochs):
            training_loss = 0
            training_state = None
            batch_data , batch_label , batch_indices = util.gen_batch(sequence , labels , params , max_len , max_indices)

            for j in range(len(batch_data)):
                x = []
                y = []
                ids = []
            
                # Fill feed_dict
                feed_dict = {graph['x']:batch_data[j], graph['y']:batch_label[j], graph['keep_prob']:0.5 , graph['sequence_length']:batch_indices[j]} # drop out prob. = 0.5
                if training_state is not None:
                    graph['init_state'] = training_state
                training_loss_, training_state, _ = \
                sess.run([graph['total_loss'],
                        graph['final_state'],
                        graph['train_step']], feed_dict=feed_dict)
                training_loss += training_loss_
            print('loss=' , training_loss/(i+1))
        print('save model to ', model_path)
        model_saver.save(sess, 'model/rnn-layer3-512.ckpt')
        return sess , training_loss

path = 'data/mfcc/'
lab_path = 'data/'
map_path = 'data/'
model_path = 'model/'


data , map48_39 , map48_char = util.read_data(path , lab_path , map_path)
tmp = np.asarray(data.values)
print(tmp.shape)
sequence , labels , max_len , max_indices = util.gen_sequence(data.values , params)
print('sequence=', np.array(sequence).shape)

#batch_data , batch_label , batch_indices = util.gen_batch(sequence , labels , params , max_len , max_indices)
graph = model_rnn.rnn_lstm(params , max_len)
# Training
sess, training_loss = train(params, sequence , labels , graph , model_path , max_len , max_indices)       


