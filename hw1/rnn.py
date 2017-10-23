import tensorflow as tf

def rnn_lstm(params , max_len):

    state_size = params['state_size']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_steps = params['num_steps']    
    feature_num = params['feature_num']
    learning_rate = params['learning_rate']

    x = tf.placeholder(tf.float32, [None, max_len, feature_num], name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, max_len], name='output_placeholder')
    keep_prob = tf.placeholder(tf.float32 , name = 'keep_prob')
    sequence_length = tf.placeholder(tf.int32 , name = 'sequence_length')

    batch_size = tf.shape(x)[0]
    # Coding ouput by one-hot encoding
    y_one_hot = tf.one_hot(y, num_classes, dtype=tf.int32)
  
    # print("x dim = {}".format(x.get_shape()))
    # print("y_one_hot dim = {}".format(y_one_hot.get_shape()))
    
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell , output_keep_prob = keep_prob)
    
    cell2 = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell2 = tf.contrib.rnn.DropoutWrapper(cell2 , output_keep_prob = keep_prob)
    
    cells = tf.nn.rnn_cell.MultiRNNCell([cell , cell2] , state_is_tuple=True)

    init_state = cell.zero_state(batch_size, tf.float32)    
    
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cells, x , dtype = tf.float32 , sequence_length=sequence_length)
        
    # print("rnn_outputs dim = {}".format(rnn_outputs.get_shape()))
    # print("final_state dim = {}".format(final_state.get_shape()))
  
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
  
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    # y_reshaped = tf.reshape(y_one_hot, [-1])
    logits = tf.reshape(tf.matmul(rnn_outputs, W) + b, [batch_size, max_len, num_classes])
    #last_frame = tf.slice(logits, [0, num_steps-1, 0], [batch_size, 1, num_classes])
    #print('last fram dim={}'.format(last_frame.get_shape()))
    predictions_one_hot = tf.nn.softmax(logits)
    print('preditions dim=', predictions_one_hot.get_shape())
    predictions = tf.argmax(predictions_one_hot, axis=1)
    print('predictions dim=', predictions.get_shape())
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
  
    # print('logits dim={}'.format(logits.get_shape()))
    # print('y_reshaped dim={}'.format(y_reshaped.get_shape())    )
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    tf.add_to_collection('predictions', predictions)
    tf.add_to_collection('predictions_one_hot', predictions_one_hot)
    tf.add_to_collection('keep_prob', keep_prob)
    tf.add_to_collection('x', x)
    

    return dict(
        x = x,
        y = y,
        accuracy = accuracy,
        keep_prob = keep_prob,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        predictions = predictions,
        sequence_length = sequence_length
    )