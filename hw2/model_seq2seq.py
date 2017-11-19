import s2vt
import util
import tensorflow as tf
import numpy as np
import os
from keras.preprocessing import sequence
import time

params = dict(
    dim_image = 4096,
    dim_hidden = 256,
    n_video_lstm_step = 80,
    n_caption_lstm_step = 40,
    n_lstm_steps = 80,
    n_epochs = 250,
    batch_size = 64,
    learning_rate = 0.001,
    keep_prob = 0.7
)
os.environ['CUDA_VISIBLE_DEVICES']='0'
train_data_path = './MLDS_hw2_data/training_data/feat'
test_data_path = './MLDS_hw2_data/testing_data/feat'

train_label_path = './MLDS_hw2_data/training_label.json'
test_label_path = './MLDS_hw2_data/testing_label.json'

model_path = './model4'

def train(params , train_data_path , train_label_path , model_path):
    
    n_epochs = params['n_epochs']
    n_video_lstm_step = params['n_video_lstm_step']
    n_caption_lstm_step = params['n_caption_lstm_step']
    dim_hidden = params['dim_hidden']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    dim_image = params['dim_image']
    keep_prob = params['keep_prob']
    
    train_captions = util.parse(train_label_path)

    captions_list = list([])
    v_ids = list([])
    
    for i, video in enumerate(train_captions):
        captions_list = captions_list + list(video['caption'])
        v_ids = v_ids + list([video['id']] * len(video['caption']))

    captions = np.asarray(captions_list, dtype=np.object)
    v_ids = np.asarray(v_ids, dtype=np.object)

    #print('captions: ' , captions)
    #print('v_ids: ' , v_ids)

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    captions = list(captions)
    captions = np.asarray(captions, dtype=np.object)

    wordtoix, ixtoword, bias_init_vector = s2vt.preProBuildWordVocab(captions, word_count_threshold=3)
    
    np.save("./wordtoix", wordtoix)
    np.save('./ixtoword', ixtoword)
    np.save("./bias_init_vector", bias_init_vector)

    n_words=len(wordtoix)
    model = s2vt.S2VT(
            params,
            n_words,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    print('tf_loss: ', tf_loss)
    sess = tf.InteractiveSession()
    
    
    saver = tf.train.Saver()
    
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()

    

    #loss_fd = open('loss.txt', 'w')
    loss_to_draw = []

    for epoch in range(1 , n_epochs+1):
        loss_to_draw_epoch = []

        index = list(range(len(captions)))
        np.random.shuffle(index)
        s_captions = captions[index]
        s_v_ids = v_ids[index]

        for start, end in zip(
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):
            print('start:' , start)
            print('end' , end)
            start_time = time.time()

            current_batch = s_captions[start:end]
            current_videos = s_v_ids[start:end]

            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            current_feats_vals = list(map(lambda vid: np.load(os.path.join(train_data_path , vid + '.npy')), current_videos))

            current_video_masks = np.zeros((batch_size, n_video_lstm_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = list(map(lambda x: '<bos> ' + x, current_batch))

            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix )) )

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1            

            _, loss_val = sess.run([train_op, tf_loss], feed_dict={
                            tf_video: current_feats,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                            })
            loss_to_draw_epoch.append(loss_val)
            """
            if start != 0:
                print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
            """
            print('idx: ', int(end / batch_size), "/", int(len(captions)/batch_size), " Epoch: ", epoch, " loss: ", loss_val, 'm loss: ', np.mean(loss_to_draw_epoch), ' Elapsed time: ', str((time.time() - start_time)))
            #loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')
        
        if np.mod(epoch, 10) == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

    #loss_fd.close()

train(params , train_data_path , train_label_path , model_path)
