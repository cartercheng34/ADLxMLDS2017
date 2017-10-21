import util
import tensorflow as tf
import numpy as np
path = 'data/mfcc/'
lab_path = 'data/'
map_path = 'data/'

params = dict(
    input_size = 28,
    hidden_size = 128,
    num_layers = 2,
    num_classes = 10,
    batch_size = 32,
    num_epochs = 50,
    num_steps = 20,
    learning_rate = 0.001,
    training_ratio = 0.9,
    feature_num = 39)

data = util.read_data(path , lab_path , map_path)

tmp = np.asarray(data.values)
print(tmp.shape)

sequence , labels = util.gen_sequence(data.values , params)
print(np.asarray(sequence).shape)
print('sequence=', np.array(sequence).shape)


batch_data , batch_label = util.gen_batch(sequence , labels , params)
print(np.array(batch_data.shape))
print(np.array(batch_label.shape))
# split
train_batch , valid_batch = util.cross_validation(batch_data , params)
print(np.asarray(train_batch).shape)
print(np.asarray(valid_batch).shape)

train_label , valid_label = util.cross_validation(batch_label , params)
print(np.asarray(train_label).shape)
print(np.asarray(valid_label).shape)
#for i in range(len(data))