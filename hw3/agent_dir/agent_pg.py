import numpy as np
import gym
import tensorflow as tf
import time
import os
from agent_dir.agent import Agent
import pickle

# tf operations

def tf_discount_rewards(tf_r , gamma): #tf_r ~ [game_steps,1]
    discount_f = lambda a, v: a*gamma + v;
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
    return tf_discounted_r

def discount_rewards(r):
    gamma = 0.99
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    # for t in reversed(range(0, r.size)):
    for t in reversed(range(0, len(r))):
      if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
      running_add = running_add * gamma + r[t]
      discounted_r[t] = running_add
    return discounted_r

def tf_policy_forward(x , tf_model): #x ~ [1,D]
    h = tf.matmul(x, tf_model['W1'])
    h = tf.nn.relu(h)
    logp = tf.matmul(h, tf_model['W2'])
    p = tf.nn.softmax(logp)
    return p

# downsampling
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        """        
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        tf.reset_default_graph()
        self.n_obs = 80 * 80
        self.hidden_units = 200
        self.n_actions = 3
        self.batch_size = 10
        self.env = env
        self.learning_rate = 1e-3

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            save_path = 'models/'
            tf_model = {}
            with tf.variable_scope('layer_one',reuse=False):
                xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.n_obs), dtype=tf.float32)
                tf_model['W1'] = tf.get_variable("W1", [self.n_obs, self.hidden_units], initializer=xavier_l1)
            with tf.variable_scope('layer_two',reuse=False):
                xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.hidden_units), dtype=tf.float32)
                tf_model['W2'] = tf.get_variable("W2", [self.hidden_units , self.n_actions], initializer=xavier_l2)
            
            # tf placeholders
            self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_obs],name="tf_x")
            self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions],name="tf_y")
            self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")

            self.tf_aprob = tf_policy_forward(self.tf_x , tf_model)
            
            self.sess = tf.InteractiveSession()

            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(save_path))
        
        ##################
        # YOUR CODE HERE #
        ##################
        
        #tf.reset_default_graph()
         


    def init_game_setting(self):
        
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        
        ##################
        # YOUR CODE HERE #
        ##################
        
               

        self.prev_x = None


    def train(self):
        
        #Implement your training algorithm here
        
        ##################
        # YOUR CODE HERE #
        ##################
        gamma = .99               # discount factor for reward
        decay = 0.99              # decay rate for RMSProp gradients
        save_path='models/pong.ckpt'

        observation = self.env.reset()
        prev_x = None
        xs,rs,ys = [],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        
        tf_model = {}
        with tf.variable_scope('layer_one',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.n_obs), dtype=tf.float32)
            tf_model['W1'] = tf.get_variable("W1", [self.n_obs, self.hidden_units], initializer=xavier_l1)
        with tf.variable_scope('layer_two',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.hidden_units), dtype=tf.float32)
            tf_model['W2'] = tf.get_variable("W2", [self.hidden_units , self.n_actions], initializer=xavier_l2)
        
        # tf placeholders
        tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_obs],name="tf_x")
        tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions],name="tf_y")
        tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")
        
        # tf reward processing (need tf_discounted_epr for policy gradient wizardry)
        tf_discounted_epr = tf_discount_rewards(tf_epr , gamma)
        tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
        tf_discounted_epr -= tf_mean
        tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)
        
        # tf optimizer op
        tf_aprob = tf_policy_forward(tf_x , tf_model)
        loss = tf.nn.l2_loss(tf_y-tf_aprob)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=decay)
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
        train_op = optimizer.apply_gradients(tf_grads)

        # tf graph initialization
        sess = tf.InteractiveSession()
        #tf.initialize_all_variables().run()
        sess.run(tf.global_variables_initializer())

        # try load saved model
        saver = tf.train.Saver(tf.all_variables())
        load_was_success = True # yes, I'm being optimistic
        try:
            save_dir = '/'.join(save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            saver.restore(sess, load_path)
        except:
            print ("no saved model to load. starting new session")
            load_was_success = False
        else:
            print ("loaded model: {}".format(load_path))
            saver = tf.train.Saver(tf.all_variables())
            episode_number = int(load_path.split('-')[-1])

        # training loop
        while True:
        #     if True: env.render()

            # preprocess the observation, set input to network to be difference image
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.n_obs)
            prev_x = cur_x

            # stochastically sample a policy from the network
            feed = {tf_x: np.reshape(x, (1,-1))}
            aprob = sess.run(tf_aprob,feed) ; aprob = aprob[0,:]
            action = np.random.choice(self.n_actions, p=aprob)
            label = np.zeros_like(aprob) ; label[action] = 1

            # step the environment and get new measurements
            observation, reward, done, info = self.env.step(action+1)
            reward_sum += reward
            
            # record game history
            xs.append(x) ; ys.append(label) ; rs.append(reward)
            
            if done:
                # update running reward
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                """
                temp_rs = np.vstack(rs)
                discount_r = discount_rewards(temp_rs)
                discount_r = discount_r - np.mean(discount_r)
                discount_r = discount_r / np.std(discount_r)
                """
                # parameter update
                feed = {tf_x: np.vstack(xs), tf_epr: np.vstack(rs), tf_y: np.vstack(ys)}
                _ = sess.run(train_op,feed)
                
                # print progress console
                if episode_number % 10 == 0:
                    print ('ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward))
                else:
                    print ('\tep {}: reward: {}'.format(episode_number, reward_sum))
                
                # bookkeeping
                xs,rs,ys = [],[],[] # reset game history
                episode_number += 1 # the Next Episode
                observation = self.env.reset() # reset env
                fp = open('pg_plot/reward.p' , 'wb')
                pickle.dump(reward_sum , fp)
                reward_sum = 0
                if episode_number % 100 == 0:
                    saver.save(sess, save_path, global_step=episode_number)
                    print ("SAVED MODEL #{}".format(episode_number))



    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        

        """
        try:
            save_dir = '/'.join(save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            saver.restore(sess, load_path)            
        except:
            print ("no saved model to load. starting new session")
            load_was_success = False
        else:
            print ("loaded model: {}".format(load_path))
            saver = tf.train.Saver(tf.all_variables())
            episode_number = int(load_path.split('-')[-1])
        """

        

        
        #if True: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.n_obs)
        self.prev_x = cur_x

        # stochastically sample a policy from the network
        feed = {self.tf_x: np.reshape(x, (1,-1))}
        aprob = self.sess.run(self.tf_aprob,feed) ; aprob = aprob[0,:]
        action = np.random.choice(self.n_actions, p=aprob)
        

        #label = np.zeros_like(aprob) ; label[action] = 1

        # step the environment and get new measurements
        #observation, reward, done, info = env.step(action+1)

        return action + 1
