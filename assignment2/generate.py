import random
import os
import numpy as np
import collections
import torch

from simple_RNN import RNN

GENERATED_SEQ_LEN = 10

MODEL_PATH = 'RNN_ADAM_model_RNN_optimizer_ADAM_initial_lr_0.0001_batch_size_20_seq_len_35_hidden_size_1500_num_layers_2_dp_keep_prob_0.35_save_best_0'
EMB_SIZE = 200
HIDDEN_SIZE = 1500
SEQ_LEN = 35
BATCH_SIZE = 20
VOCAB_SIZE = 10000
NUM_LAYERS = 2
DP_KEEP_PROB = 0.35


# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

word_to_id, id_2_word = _build_vocab(os.path.join('data', 'ptb' + ".train.txt"))

load_path = os.path.join(MODEL_PATH, 'best_params.pt')

model_class = RNN if MODEL_PATH.split('_')[0] == 'RNN' else GRU

model = model_class(emb_size=EMB_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    seq_len=SEQ_LEN,
                    batch_size=BATCH_SIZE,
                    vocab_size=VOCAB_SIZE,
                    num_layers=NUM_LAYERS,
                    dp_keep_prob=DP_KEEP_PROB)

model.load_state_dict(torch.load(load_path))
hidden = model.init_hidden()

hidden = hidden.to(device)
model = model.to(device)
model.eval()

# Random first input
_input = torch.zeros([1, 1], dtype=torch.long)
rand_int = random.randint(0, len(word_to_id))
_input[0, 0] = rand_int

seq = model.generate(_input, hidden, generated_seq_len=GENERATED_SEQ_LEN, device=device)

print(MODEL_PATH)
print('>>> first word: \n {}'.format(id_2_word[rand_int]))
print('>>> generated seq: \n {}'.format(' '.join([id_2_word[int(i)] for i in seq])))
