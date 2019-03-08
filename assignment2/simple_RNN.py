import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# TODO: Add dropout after embedding https://ift6135forum.slack.com/archives/CGF0C0C4U/p1551981909143800?thread_ts=1551936476.131800&cid=CGF0C0C4U
# TODO: have different input size (emb_size or hidden_size) depending if its the first recurrent layer

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is
# what the main script expects. If you modify the contract,
# you must justify that choice, note it in your report, and notify the TAs
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.


def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers=2, dp_keep_prob=0.5):

    """
    emb_size:     The numvwe of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.vocab_size = vocab_size

    # Embedding encoder
    self.embedding = nn.Embedding(vocab_size, emb_size)

    # N stacked recurrent layers (first layer has different input size)
    linear_W = [nn.Linear(emb_size, hidden_size)] + \
               [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
    self.linear_W = nn.ModuleList(linear_W)
    self.linear_U = clones(nn.Linear(hidden_size, hidden_size), num_layers)

    # Embedding decoder
    self.decode = nn.Linear(hidden_size, vocab_size)

    self.drop_prob = 1 - dp_keep_prob
    self.activation = nn.Tanh()
    self.softmax = nn.Softmax(dim=2)

    # Weight initialization (Embedding has no bias)
    self.init_weights_uniform(self.embedding, init_bias=False)
    self.init_weights_uniform(self.decode, init_bias=True)

  def init_weights_uniform(self, layer, init_bias=False):
    """
    Initialize all the weights uniformly in the range [-0.1, 0.1]
    and all the biases to 0 (in place)
    """
    torch.nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
    if init_bias:
        torch.nn.init.constant_(layer.bias, 0)

  def init_hidden(self):
    """
    This is used for the first mini-batch in an epoch, only.
    """
    return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)

    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the
              mini-batches in an epoch, except for the first, where the return
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details,
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    h_previous_ts = hidden
    logits = []
    emb = self.embedding(inputs)
    for i in range(self.seq_len):
        h_next_ts = []
        h_previous_layer = nn.functional.dropout(emb[i], self.drop_prob)
        for l in range(self.num_layers):
            # Recurrent layer
            a_W = self.linear_W[l](h_previous_layer)
            a_U = self.linear_U[l](h_previous_ts[l])
            h_recurrent = self.activation(a_U + a_W)
            # Fully connected layer
            h_previous_layer = nn.functional.dropout(h_recurrent, self.drop_prob)
            # Keep the ref for next ts
            h_next_ts.append(h_recurrent)
        h_previous_ts = torch.stack(h_next_ts)
        logits.append(self.decode(h_previous_layer))
    return torch.stack(logits), h_previous_ts

  def generate(self, input, hidden, generated_seq_len):
    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """

    self.eval()

    samples = []
    h_previous_ts = hidden
    h_next_ts = []

    new_input = input

    for i in range(generated_seq_len):
        new_input = new_input.to(torch.device("cuda"))
        emb = self.embedding(new_input)
        h_previous_layer = emb
        for l in range(self.num_layers):
            # Recurrent layer
            a_W = self.linear_W[l](h_previous_layer)
            a_U = self.linear_U[l](h_previous_ts[l])
            h_recurrent = self.activation(a_U + a_W)
            # Fully connected layer
            h_previous_layer = self.dropout(self.fc[l](h_recurrent))
            # Keep the ref for next ts
            h_next_ts.append(h_recurrent)

        h_previous_ts = torch.stack(h_next_ts)

        sample = h_previous_layer
        sample = self.softmax(self.decode(sample))
        sample_index = int(np.argmax(sample.cpu().detach().numpy()))
        samples.append(sample_index)
        new_input[0, 0] = sample_index

    return samples
