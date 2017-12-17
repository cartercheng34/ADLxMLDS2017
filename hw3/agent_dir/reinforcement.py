import tensorflow as tf
import numpy as np

class PolicyGradient:

    def __init__(self, env):
        #self.n_actions = env.action_space.n
        self.n_actions = 3
        self.learning_rate = 0.001
        self.batch_size = 5
        self.frame_index = 0

        self.build_net()

        self.reset_episodes()

        self.sess  = tf.Session()
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        '''
        Build the policy network for choosing action.
        '''
        self.tf_states  = tf.placeholder(tf.float32, [None, 6400], name='states')
        self.tf_actions = tf.placeholder(tf.int32  , [None], name='actions')
        self.tf_values  = tf.placeholder(tf.float32, [None], name='values')

        init = tf.truncated_normal_initializer(0, 0.02)

        def conv2d_layer(input_tensor, filter_shape, strides, name, padding='SAME', activation=tf.nn.relu):
            with tf.variable_scope(name, initializer=init):
                filters = tf.get_variable('filters', filter_shape, dtype=tf.float32)
                bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32)
                conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
                output = activation(tf.nn.bias_add(conv, bias))
                return output
        
        def dense_layer(input_tensor, input_dim, output_dim, name, activation=tf.nn.relu):
            with tf.variable_scope(name, initializer=init):
                W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32)
                #b = tf.get_variable('bias', [output_dim], dtype=tf.float32)
                output = activation(tf.matmul(input_tensor, W))
                return output

        def max_pool_layer(input_tensor, ksize, strides, padding='SAME'):
            pool = tf.nn.max_pool(input_tensor, ksize, strides, padding)
            return pool

        '''
        conv1 = conv2d_layer(input_tensor=self.tf_states, 
                             filter_shape=[5, 5, 3, 32],
                             strides=[1, 1, 1, 1],
                             name='conv1')
        pool1 = max_pool_layer(input_tensor=conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1])
        flatten_dim = np.prod(pool1.get_shape().as_list()[1:])
        flatten = tf.reshape(pool1, [-1, flatten_dim])
        '''
        dense2 = dense_layer(self.tf_states, 6400, 256, 'dense2')
        self.action_logits = dense_layer(dense2, 256, self.n_actions, 'logits', activation=tf.identity)
        self.action_probs = tf.nn.softmax(self.action_logits, dim=-1)
        
        '''
        loss = tf.nn.l2_loss(tf.one_hot(self.tf_actions, self.n_actions) - self.action_probs)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99)
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf.expand_dims(self.tf_values, -1))
        self.train_step = optimizer.apply_gradients(tf_grads)
        '''
        neg_log_probs = tf.reduce_sum(-tf.log(self.action_probs + 1e-10) * tf.one_hot(self.tf_actions, self.n_actions), axis=1)
        loss = tf.reduce_mean(neg_log_probs * self.tf_values)
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)

    def load_model(self, model_path):
        '''
        Load pretrained model from file.
        '''
        self.saver.restore(self.sess, model_path)

    def store_step(self, state, prev_state, action, reward, done):
        '''
        Store informations of a step into the episode history.
        '''
        state = self.preprocess(state)
        prev_state = self.preprocess(prev_state)
        self.states.append(state - prev_state)
        self.actions.append(action)
        self.rewards.append(reward)
        if done:
            self.values.append(self.get_value(self.rewards[self.frame_index:]))
            self.frame_index = len(self.rewards)
        '''
        state = self.preprocess(state)
        self.states  = np.append(self.states, np.expand_dims(state, 0), axis=0)
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, reward)
        if done:
            self.values = np.append(self.values, self.get_value(self.rewards[self.frame_index:]))
            self.frame_index = len(self.rewards)
        '''

    def reset_episodes(self):
        '''
        Clear and reset the episode history.
        '''
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.frame_index = 0
        '''
        self.states  = np.array([], dtype=np.uint8).reshape(0, 6400)
        self.actions = np.array([])
        self.rewards = np.array([])
        self.values  = np.array([])
        self.frame_index = 0
        '''

    def learn(self):
        '''
        Update the policy network parameters.
        '''
        self.states  = np.array(self.states)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.values  = np.concatenate(self.values)
        self.actions -= 1 
        self.sess.run(self.train_step, feed_dict={self.tf_states:  self.states,
                                                  self.tf_actions: self.actions,
                                                  self.tf_values:  self.values})

    def get_value(self, rewards, gamma=0.99):
        '''
        Return a discounted value function array according to the rewards array.
        '''
        r_sum = 0 
        values = np.zeros(len(rewards))
        for i in reversed(range(len(rewards))):
            if rewards[i] != 0: r_sum = 0
            r_sum = rewards[i] + r_sum * gamma
            values[i] = r_sum
        values = (values - values.mean()) / values.std()
        return values

    def choose_action(self, state, prev_state):
        '''
        Choose an action given provided state.

        Actions:  0,1: Stay
                  2,4: Up
                  3,5: Down 
        '''
        prev_state = self.preprocess(prev_state)
        state = self.preprocess(state)
        x = np.expand_dims(state - prev_state, 0)
        action_probs = self.sess.run(self.action_probs, feed_dict={self.tf_states: x})
        action = np.random.choice(self.n_actions, p=action_probs.reshape(-1))
        return action + 1  # choose from 1,2,3

    def save_model(self, file_path):
        '''
        Save current model to file.
        '''
        self.saver.save(self.sess, file_path)

    def preprocess(self, img):
        '''
        Image preprocess. Crop, resize, and remove background.
        '''
        state = img[34:194, :, 0]  
        state = state[::2, ::2]
        state[state == 144] = 0
        state[state == 109] = 0
        state[state != 0] = 1
        return state.reshape(-1)

class ActorCritic:

    def __init__(self, env):
        #self.n_actions = env.action_space.n
        self.n_actions = 3
        self.actor_lr  = 0.001
        self.critic_lr = 0.01
        self.frame_index = 0
        self.gamma = 0.95

        self.build_net()

        self.sess  = tf.Session()
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        '''
        Build the policy network for choosing action.
        '''
        init = tf.truncated_normal_initializer(0, 0.02)

        def conv2d_layer(input_tensor, filter_shape, strides, name, padding='SAME', activation=tf.nn.relu):
            with tf.variable_scope(name, initializer=init):
                filters = tf.get_variable('filters', filter_shape, dtype=tf.float32)
                bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32)
                conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
                output = activation(tf.nn.bias_add(conv, bias))
                return output
        
        def dense_layer(input_tensor, input_dim, output_dim, name, activation=tf.nn.relu):
            with tf.variable_scope(name, initializer=init):
                W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32)
                #b = tf.get_variable('bias', [output_dim], dtype=tf.float32)
                output = activation(tf.matmul(input_tensor, W))
                return output

        def max_pool_layer(input_tensor, ksize, strides, padding='SAME'):
            pool = tf.nn.max_pool(input_tensor, ksize, strides, padding)
            return pool

        ############################# Actor ################################
        self.actor_state  = tf.placeholder(tf.float32, [1, 6400], name='actor_state')
        self.actor_action = tf.placeholder(tf.int32  , None, name='action')
        self.actor_tderr  = tf.placeholder(tf.float32, None, name='td_error')

        with tf.variable_scope('actor'):
            dense = dense_layer(self.actor_state, 6400, 256, 'dense')
            self.action_logits = dense_layer(dense, 256, self.n_actions, 'logits', activation=tf.identity)
            self.action_probs = tf.nn.softmax(self.action_logits, dim=-1)
            
            neg_log_prob = -tf.log(self.action_probs[0, self.actor_action] + 1e-10)
            loss = tf.reduce_mean(neg_log_prob * self.actor_tderr)
            self.actor_train_step = tf.train.RMSPropOptimizer(self.actor_lr).minimize(loss)

        ############################# Critic ################################
        self.critic_state = tf.placeholder(tf.float32, [1, 6400], name='actor_state')
        self.critic_next_value = tf.placeholder(tf.int32, None, name='next_value')
        self.critic_reward = tf.placeholder(tf.float32, None, name='reward')

        with tf.variable_scope('critic'):
            dense = dense_layer(self.critic_state, 6400, 256, 'dense')
            self.critic_out_value = dense_layer(dense, 256, 1, 'output_value', tf.identity)

            self.td_error = self.critic_reward + self.gamma * self.critic_next_value - self.critic_out_value
            loss = tf.square(self.td_error)
            self.critic_train_step = tf.train.RMSPropOptimizer(self.critic_lr).minimize(loss)

    def save_model(self, file_path, global_step):
        '''
        Save current model to file.
        '''
        self.saver.save(self.sess, file_path, global_step=global_step)

    def load_model(self, model_path):
        '''
        Load pretrained model from file.
        '''
        self.saver.restore(self.sess, model_path)

    def learn(self, state, action, reward, next_state):
        '''
        Update the policy network parameters.
        '''
        state = np.expand_dims(self.preprocess(state), 0)
        next_state = np.expand_dims(self.preprocess(next_state), 0)
        action -= 1 

        v_next = self.sess.run(self.critic_out_value, feed_dict={self.critic_state: next_state})
        td_error, _ = self.sess.run([self.td_error, self.critic_train_step], feed_dict={self.critic_state: state,
                                                                                        self.critic_next_value: v_next,
                                                                                        self.critic_reward: reward})
        self.sess.run(self.actor_train_step, feed_dict={self.actor_state:  state,
                                                        self.actor_action: action,
                                                        self.actor_tderr:  td_error})

    def choose_action(self, state):
        '''
        Choose an action given provided state.

        Actions:  0,1: Stay
                  2,4: Up
                  3,5: Down 
        '''
        state = self.preprocess(state)
        x = np.expand_dims(state, 0)
        action_probs = self.sess.run(self.action_probs, feed_dict={self.actor_state: x})
        action = np.random.choice(self.n_actions, p=action_probs.reshape(-1))
        return action + 1  # choose from 1,2,3

    def preprocess(self, img):
        '''
        Image preprocess. Crop, resize, and remove background.
        '''
        state = img[34:194, :, 0]  
        state = state[::2, ::2]
        state[state == 144] = 0
        state[state == 109] = 0
        state[state != 0] = 1
        return state.reshape(-1)

class DQN:

    def __init__(self, env, test=False):
        self.n_actions = env.action_space.n
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.batch_size = 32
        self.memory_size = 10000
        self.memory_count = 0
        self.learn_step_count = 0
        self.update_target_freq = 250
        if test:
            self.epsilon = 0
            self.epsilon_decay = 0
            self.epsilon_min = 0
        else:
            self.epsilon = 1
            self.epsilon_decay = 1e-6
            self.epsilon_min = 0.05

        self.build_net()

        self.reset_memory()

        self.sess  = tf.Session()
        self.saver = tf.train.Saver()

        q_params = tf.get_collection('Q_network')
        t_params = tf.get_collection('target_network')
        self.update_target_op = [tf.assign(t, q) for t, q in zip(t_params, q_params)]

        self.sess.run(tf.global_variables_initializer())
        self.loss_record = []

    def build_net(self):

        def conv2d_layer(input_tensor, filter_shape, strides, name, collections, padding='SAME', activation=tf.nn.relu):
            with tf.variable_scope(name):
                filters = tf.get_variable('filters', filter_shape, dtype=tf.float32, collections=collections)
                bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32, collections=collections)
                conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
                output = activation(tf.nn.bias_add(conv, bias))
                return output
        
        def dense_layer(input_tensor, input_dim, output_dim, name, collections, activation=tf.nn.relu):
            with tf.variable_scope(name):
                W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32, collections=collections)
                b = tf.get_variable('bias', [output_dim], dtype=tf.float32, collections=collections)
                output = activation(tf.matmul(input_tensor, W))
                return output

        def lrelu(x, alpha=0.2):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

        ####################### Q Network ##########################
        self.tf_states = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states')

        with tf.variable_scope('Q_network'):
            collection_name = ['Q_network', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = conv2d_layer(self.tf_states, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
            conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
            conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
            flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
            flatten = tf.reshape(conv3, [-1, flatten_dim])
            dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
            self.Q_value = dense_layer(dense, 512, self.n_actions, 'output', collection_name, tf.identity)

        #################### Target Q Network ######################
        self.tf_states_t = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states_target')

        with tf.variable_scope('target_network'):
            collection_name = ['target_network', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = conv2d_layer(self.tf_states_t, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
            conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
            conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
            flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
            flatten = tf.reshape(conv3, [-1, flatten_dim])
            dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
            self.Q_target = dense_layer(dense, 512, self.n_actions, 'output', collection_name, tf.identity)

        ################### Loss and Optimizer #####################
        self.tf_actions = tf.placeholder(tf.int32, [None], name='actions')
        self.tf_y = tf.placeholder(tf.float32, [None], name='y')

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, tf.one_hot(self.tf_actions, self.n_actions)), axis=-1)
        self.loss = tf.reduce_mean(tf.squared_difference(self.tf_y, Q_action))
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def learn(self):
        '''
        Update parameters.
        '''
        # Update target network
        if self.learn_step_count % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        # Sample batch from memory
        if self.memory_count > self.memory_size:
            indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_count, size=self.batch_size)
        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_state_batch = self.next_states[indices]
        done_batch = self.done[indices]

        # Calculate Target Q values
        Q_target = self.sess.run(self.Q_target, feed_dict={self.tf_states_t: next_state_batch})

        # Calculate y
        y_batch = [reward_batch[i] if done_batch[i] else reward_batch[i] + self.gamma * np.max(Q_target[i]) 
            for i in range(self.batch_size)]

        # Update parameters
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.tf_states: state_batch,
                                                                         self.tf_actions: action_batch,
                                                                         self.tf_y: y_batch})
        self.loss_record.append(loss)
        self.learn_step_count += 1

    def reset_memory(self):
        '''
        Reset the transition memories.
        '''
        self.states = np.zeros((self.memory_size, 84, 84, 4))
        self.actions = np.zeros(self.memory_size)
        self.rewards = np.zeros(self.memory_size)
        self.next_states = np.zeros((self.memory_size, 84, 84, 4))
        self.done = np.zeros(self.memory_size)

    def store_step(self, state, action, reward, next_state, done):
        '''
        Store a transition into the memory.
        '''
        index = self.memory_count % self.memory_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.done[index] = done
        self.memory_count += 1

    def choose_action(self, state):
        '''
        Choose an action given provided state.
        '''
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action_values = self.sess.run(self.Q_value, feed_dict={self.tf_states: np.expand_dims(state, 0)})
            action = np.argmax(action_values)
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        return action

    def save_model(self, file_path, step):
        '''
        Save current model to file.
        '''
        self.saver.save(self.sess, file_path, global_step=step)

    def load_model(self, model_path):
        '''
        Load pretrained model from file.
        '''
        self.saver.restore(self.sess, model_path)

class DoubleDQN(DQN):

    def __init__(self, env, test=False):
        super(DoubleDQN,self).__init__(env, test)

    def learn(self):
        '''
        Update parameters.
        '''
        # Update target network
        if self.learn_step_count % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        # Sample batch from memory
        if self.memory_count > self.memory_size:
            indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_count, size=self.batch_size)
        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_state_batch = self.next_states[indices]
        done_batch = self.done[indices]

        # Calculate Target Q values
        Q_target, Q_value_next = self.sess.run([self.Q_target, self.Q_value], feed_dict={self.tf_states_t: next_state_batch,
                                                                                         self.tf_states: next_state_batch})

        # Calculate y
        y_batch = [reward_batch[i] if done_batch[i] else reward_batch[i] + self.gamma * Q_target[i][np.argmax(Q_value_next[i])] 
            for i in range(self.batch_size)]

        # Update parameters
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.tf_states: state_batch,
                                                                         self.tf_actions: action_batch,
                                                                         self.tf_y: y_batch})
        self.loss_record.append(loss)
        self.learn_step_count += 1

class DuelingDQN(DQN):

    def __init__(self, env, test=False):
        super(DuelingDQN,self).__init__(env, test)

    def build_net(self):

        def conv2d_layer(input_tensor, filter_shape, strides, name, collections, padding='SAME', activation=tf.nn.relu):
            with tf.variable_scope(name):
                filters = tf.get_variable('filters', filter_shape, dtype=tf.float32, collections=collections)
                bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32, collections=collections)
                conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
                output = activation(tf.nn.bias_add(conv, bias))
                return output
        
        def dense_layer(input_tensor, input_dim, output_dim, name, collections, activation=tf.nn.relu):
            with tf.variable_scope(name):
                W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32, collections=collections)
                b = tf.get_variable('bias', [output_dim], dtype=tf.float32, collections=collections)
                output = activation(tf.matmul(input_tensor, W))
                return output

        def lrelu(x, alpha=0.2):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

        ####################### Q Network ##########################
        self.tf_states = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states')

        with tf.variable_scope('Q_network'):
            collection_name = ['Q_network', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = conv2d_layer(self.tf_states, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
            conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
            conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
            flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
            flatten = tf.reshape(conv3, [-1, flatten_dim])
            dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
            V = dense_layer(dense, 512, 1, 'value', collection_name, tf.identity)
            A = dense_layer(dense, 512, self.n_actions, 'advantage', collection_name, tf.identity)
            self.Q_value = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))

        #################### Target Q Network ######################
        self.tf_states_t = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states_target')

        with tf.variable_scope('target_network'):
            collection_name = ['target_network', tf.GraphKeys.GLOBAL_VARIABLES]
            conv1 = conv2d_layer(self.tf_states_t, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
            conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
            conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
            flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
            flatten = tf.reshape(conv3, [-1, flatten_dim])
            dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
            V = dense_layer(dense, 512, 1, 'value', collection_name, tf.identity)
            A = dense_layer(dense, 512, self.n_actions, 'advantage', collection_name, tf.identity)
            self.Q_target = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))

        ################### Loss and Optimizer #####################
        self.tf_actions = tf.placeholder(tf.int32, [None], name='actions')
        self.tf_y = tf.placeholder(tf.float32, [None], name='y')

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, tf.one_hot(self.tf_actions, self.n_actions)), axis=-1)
        self.loss = tf.reduce_mean(tf.squared_difference(self.tf_y, Q_action))
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
