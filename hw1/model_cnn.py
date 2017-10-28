import tensorflow as tf


def cnn_lstm(params , max_len):
    
    # for rnn
    state_size = params['state_size']
    num_classes = params['num_classes']
    batch_size = params['batch_size']        
    feature_num = params['feature_num']
    learning_rate = params['learning_rate']
    
    #for cnn
    cnn_filter_num = params['cnn_filter_num'] # [32 , 32]
    cnn_filter_size = params['cnn_filter_size'] # 3
    cnn_pool_size = params['cnn_pool_size'] #[2,1]
    cnn_fc_layer_size = params['cnn_fc_layer_size']
    #build convolutional layer
    
    x = tf.placeholder(tf.float32, [None, max_len, feature_num], name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, max_len], name='output_placeholder')
    keep_prob = tf.placeholder(tf.float32 , name = 'keep_prob')
    sequence_length = tf.placeholder(tf.int32 , name = 'sequence_length')
    tmp_x = tf.reshape(x , [-1 , max_len , int(feature_num / 3) , 3])
    # tmp_x = tf.expand_dims(x , 3)

    input_channel = 3
    output_channel = cnn_filter_num

    filter1_shape = [cnn_filter_size , cnn_filter_size , input_channel , output_channel[0]]
    w_conv1 = tf.Variable(tf.truncated_normal(filter1_shape, stddev=0.1), name='w_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[output_channel[0]]) , name='b_conv1')
    conv1 = tf.nn.conv2d(tmp_x, w_conv1 , strides=[1, 1, 1, 1] , padding='SAME')
    pool1 = tf.nn.max_pool(tf.nn.relu(conv1 + b_conv1),\
                           ksize=[1 , cnn_pool_size[0] , cnn_pool_size[0] , 1],\
                           strides=[1 , 1 , cnn_pool_size[0] , 1],\
                           padding='SAME')

    filter2_shape = [cnn_filter_size , cnn_filter_size , output_channel[0] , output_channel[1]]
    w_conv2 = tf.Variable(tf.truncated_normal(filter2_shape, stddev=0.1), name='w_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[output_channel[1]]) , name='b_conv2')
    conv2 = tf.nn.conv2d(pool1, w_conv2 , strides=[1, 1, 1, 1] , padding='SAME')
    pool2 = tf.nn.max_pool(tf.nn.relu(conv2 + b_conv2),\
                           ksize=[1 , cnn_pool_size[1] , cnn_pool_size[1] , 1],\
                           strides=[1 , 1 , cnn_pool_size[1] , 1],\
                           padding='SAME')
    conv_output = pool2
    conv_output_dims = conv_output.get_shape().as_list() # get tensor shape , max_len x reduce feature x cnn_filter_num[-1]
    print('conv_output_dim:' , conv_output_dims)# none x max_len x reduce feature x cnn_filter_num[-1] 
    fc1_input = tf.reshape(conv_output , [-1 , conv_output_dims[1],\
                            conv_output_dims[2]*conv_output_dims[3]])
       
    

    #build fully-connected layer
    fc1_input = tf.reshape(fc1_input , [-1 , fc1_input.get_shape().as_list()[-1]]) # fc_batch x flatten_vec
    print('FC {}. Input tensor shape = {}'.format(0, fc1_input.get_shape().as_list()))
    w_fc1 = tf.Variable(tf.truncated_normal([fc1_input.get_shape().as_list()[-1] , cnn_fc_layer_size[0]], stddev=0.1) , name ='w_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[cnn_fc_layer_size[0]]) , name='b_fc1')
    fc1_output = tf.nn.relu(tf.matmul(fc1_input, w_fc1) + b_fc1)
    fc1_output = tf.nn.dropout(fc1_output , keep_prob)

    #fc2_input = tf.reshape(conv_output , [-1 , conv_output_dims[0] * conv_output_dims[1]]) # fc_batch x flatten_vec
    w_fc2 = tf.Variable(tf.truncated_normal([fc1_output.get_shape().as_list()[-1] , cnn_fc_layer_size[1]], stddev=0.1) , name ='w_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[cnn_fc_layer_size[1]]) , name='b_fc2')
    fc2_output = tf.nn.relu(tf.matmul(fc1_output, w_fc2) + b_fc2)
    fc2_output = tf.nn.dropout(fc2_output , keep_prob)

    print('fc_output_dim:' , fc2_output.get_shape().as_list()) #

    #rnn-lstm
    rnn_input = tf.reshape(fc2_output , [-1 , max_len , cnn_fc_layer_size[-1]])

    batch_size = tf.shape(rnn_input)[0]
    # Coding ouput by one-hot encoding
    y_one_hot = tf.one_hot(y, num_classes, dtype=tf.int32)
  
    # print("x dim = {}".format(x.get_shape()))
    # print("y_one_hot dim = {}".format(y_one_hot.get_shape()))
    
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell , output_keep_prob = keep_prob)
    
    cell2 = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell2 = tf.contrib.rnn.DropoutWrapper(cell2 , output_keep_prob = keep_prob)

    cell3 = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell3 = tf.contrib.rnn.DropoutWrapper(cell3 , output_keep_prob = keep_prob)
    
    cells = tf.nn.rnn_cell.MultiRNNCell([cell , cell2 , cell3] , state_is_tuple=True)

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
    predictions = tf.argmax(predictions_one_hot, axis=2)
    print('predictions dim=', predictions.get_shape())
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
  
    # print('logits dim={}'.format(logits.get_shape()))
    # print('y_reshaped dim={}'.format(y_reshaped.get_shape()))
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

    
