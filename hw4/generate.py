import tensorflow as tf
import os
import DCGAN
import skimage.io
import skimage.transform
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import numpy as np
import scipy.spatial.distance as sd
import pdb
import random
import sys

VOCAB_FILE = "./skip_thoughts/unidirectional/vocab.txt"
EMBEDDING_MATRIX_FILE = "./skip_thoughts/unidirectional/embeddings.npy"
CHECKPOINT_PATH = "./skip_thoughts/unidirectional/model.ckpt-501424"


num_epochs = 500
learning_rate = 0.0001
resume_model = False
image_dir = 'faces/'
trim_tags = 'trim.txt'
dis_updates = 1
gen_updates = 2
num_images = 5
model_path = ['new/-500' , 'new/-495' , 'new/-490' , 'new/-485' , 'new/-460']
pic_save_path = 'samples/'
early_text = 'green hair blue eyes'
#test_txt = 'sample_testing_text.txt'
test_txt = sys.argv[1]

params = dict(
    z_dim = 400,
    t_dim = 256,
    batch_size = 5,
    image_size = 64,
    gf_dim = 64,
    df_dim = 64,
    gfc_dim = 1024,
    dfc_dim = 1024,
    istrain = False,
    caption_length = 2400
)

np.random.seed(10)
gan = DCGAN.GAN(params)
_, _, _, _ = gan.build_model()

for p in range(5):

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess , model_path[p])

    input_tensors, outputs = gan.build_generator()

    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(),
                    vocabulary_file=VOCAB_FILE,
                    embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                    checkpoint_path=CHECKPOINT_PATH)



    ids = []
    tags = []

    f = open(test_txt , 'r')
    for line in f.readlines():
        id = line.split(',')[0]
        tag = line.split(',')[1]
        ids.append(id)
        tags.append(tag)

    z_noise = np.random.uniform(-1, 1, [len(ids), num_images, params['z_dim']])
    encodings = encoder.encode(tags)
    encodings = np.array(encodings)
    print(np.shape(encodings))
    encodings = np.expand_dims(encodings , axis=1)
    print(np.shape(encodings))
    caption = np.tile(encodings, (1,num_images,1))
    print(np.shape(caption))



    #caption = np.reshape(caption , (-1, num_images , params['caption_length']))

    caption_images = {}
    for i in range(0,len(ids)):
        caption_images[ids[i]] =\
        sess.run([outputs['generator']],
                feed_dict = {input_tensors['t_real_caption'] : caption[i],
                            input_tensors['t_z'] : z_noise[i]}) # (1,num_images,64,64,3)

    #pdb.set_trace()

    for i in range(len(ids)):
        #for j in range(num_images):
        skimage.io.imsave(os.path.join(pic_save_path , 'sample3_' + ids[i] + '_' + str(p) + '.jpg') , caption_images[ids[i]][0][p])
"""
encodings = encoder.encode([early_text])
z_noise = np.random.uniform(-1, 1, [1, params['z_dim']])
caption = np.tile(np.array(encodings) , (1 , 1))

[gen_image] =\
    sess.run([outputs['generator']],
            feed_dict = {input_tensors['t_real_caption'] : caption,
                        input_tensors['t_z'] : z_noise})

for i in range(1):
    skimage.io.imsave(os.path.join(pic_save_path , str(i)+ 'green_800' + '.jpg') , gen_image[i])

"""