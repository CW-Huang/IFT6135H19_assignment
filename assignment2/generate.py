import random
import os
import numpy as np
import collections
import torch

from simple_RNN import RNN

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

train_path = os.path.join('data', 'ptb' + ".train.txt")
word_to_id, id_2_word = _build_vocab(train_path)

PATH = 'RNN_SGD_LR_SCHEDULE_model_RNN_save_best_11/best_params.pt'

model = RNN(emb_size=200,
            hidden_size=200,
            seq_len=35,
            batch_size=20,
            vocab_size=10000,
            num_layers=2,
            dp_keep_prob=0.1)
model.load_state_dict(torch.load(PATH))
hidden = model.init_hidden()
hidden = hidden.to(torch.device("cuda"))
model = model.to(torch.device("cuda"))
model.eval()

# TODO: Change for batch
_input = torch.zeros([1, 1], dtype=torch.long)
rand_int = random.randint(0, len(word_to_id))
_input[0, 0] = rand_int

seq = model.generate(_input, hidden, generated_seq_len=10)

print('>>> first word: \n {}'.format(id_2_word[rand_int]))
print('>>> generated seq: \n {}'.format(' '.join([id_2_word[int(i)] for i in seq])))



