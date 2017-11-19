import s2vt
import util
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys


train_data_path = sys.argv[1] + 'training_data/feat'
#test_data_path = './MLDS_hw2_data/testing_data/feat'
test_data_path = sys.argv[1] + 'testing_data/feat'
peer_data_path = sys.argv[1] + 'peer_review/feat'

train_label_path = sys.argv[1] + 'training_label.json'
test_label_path = sys.argv[1] + 'testing_label.json'
peer_id_path = sys.argv[1] + 'peer_review_id.txt'


model_path = './model'
output_file = sys.argv[2]
peer_output = sys.argv[3]

captions = util.parse(train_label_path)
print(captions[0])
print('type:' , type(captions))

params = dict(
    dim_image = 4096,
    dim_hidden = 256,
    n_video_lstm_step = 80,
    n_caption_lstm_step = 40,
    n_lstm_steps = 80,
    n_epochs = 250,
    batch_size = 64,
    learning_rate = 0.001,
    keep_prob = 1
)

def predict(params , test_data_path , test_label_path , output_file , peer_id_path  , peer_data_path , peer_output):
    n_epochs = params['n_epochs']
    n_video_lstm_step = params['n_video_lstm_step']
    n_caption_lstm_step = params['n_caption_lstm_step']
    dim_hidden = params['dim_hidden']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    dim_image = params['dim_image']

    test_captions = util.parse(test_label_path)
    peer_ids = util.parse_peer_id(peer_id_path)
    print('ids=' , peer_ids)
    test_videos = []
    for video in test_captions:
        test_videos.append(video['id'])

    ixtoword = pd.Series(np.load('./ixtoword.npy').tolist())
 
    bias_init_vector = np.load('./bias_init_vector.npy')
    n_words=len(ixtoword)
    model = s2vt.S2VT(
            params,
            n_words,
            bias_init_vector = bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator() 
    sess = tf.InteractiveSession()
 
    saver = tf.train.Saver()
    saver.restore(sess, 'model/model-230')
    
    with open(output_file , 'w') as f:
        for idx, video_feat_path in enumerate(test_videos):
            print(idx, video_feat_path)

            video_feat = np.load(os.path.join(test_data_path, video_feat_path + '.npy'))[None,...]  
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
            

            generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            generated_words = ixtoword[generated_word_index]

            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
            generated_words = generated_words[:punctuation]

            generated_sentence = ' '.join(generated_words)
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')
            f.write(video_feat_path + ',')
            f.write(generated_sentence + '\n')

    with open(peer_output , 'w') as p:
        for i in range(len(peer_ids)):
            video_feat = np.load(os.path.join(peer_data_path, peer_ids[i] + '.npy'))[None,...] 
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
            

            generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            generated_words = ixtoword[generated_word_index]

            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
            generated_words = generated_words[:punctuation]

            generated_sentence = ' '.join(generated_words)
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')
            p.write(peer_ids[i] + ',')
            p.write(generated_sentence + '\n')


        

    
predict(params , test_data_path , test_label_path , output_file , peer_id_path  , peer_data_path , peer_output)
