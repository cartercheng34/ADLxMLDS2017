import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

VOCAB_FILE = "./skip_thoughts/unidirectional/vocab.txt"
EMBEDDING_MATRIX_FILE = "./skip_thoughts/unidirectional/embeddings.npy"
CHECKPOINT_PATH = "./skip_thoughts/unidirectional/model.ckpt-501424"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_DIR = "/dir/containing/mr/data"

text = 'short hair blue eyes blue hair'




encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

encodings = encoder.encode([text])
print(encodings)
print(np.shape(np.array(encodings)))
