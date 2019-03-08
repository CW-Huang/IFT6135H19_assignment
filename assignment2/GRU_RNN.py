import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt


def clones(module, N):
  "A helper function for producing N identical layers (each with their own parameters)."
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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


class GRU_cell(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size):
    super(GRU_cell, self).__init__()
    self.emb_size = emb_size
    self.hidden_size = hidden_size

    self.W_x = nn.Parameter(torch.Tensor(emb_size,3*hidden_size))
    self.U_h = nn.Parameter(torch.Tensor(hidden_size,2*hidden_size))
    self.U_h_tilde = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
    self.bias_rzh = nn.Parameter(torch.Tensor(3*hidden_size))

    self.init_weights_uniform()
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

  def init_weights_uniform(self):
    # Initialize all the weights uniformly in the range [-0.1, 0.1]
    # and all the biases to 0 (in place)
    stdv = 0.1
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

    torch.nn.init.zeros_(self.bias_rzh)

  def forward(self, inputs, hidden):
    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that
                    represent the index of the current token(s) in the vocabulary.
                        shape: (batch_size, embedding)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (batch_size, hidden_size)
    """
    batch_size = hidden.size(0)

    bias_rzh_batch = self.bias_rzh.unsqueeze(0).expand(batch_size, self.bias_rzh.size(0))
    W_x = torch.addmm(bias_rzh_batch,inputs,self.W_x)
    U_h_prev = torch.mm(hidden,self.U_h)
    W_rx, W_zx, W_hx = torch.split(W_x,self.hidden_size, dim=1)
    U_rh, U_zh = torch.split(U_h_prev,self.hidden_size, dim=1)

    r = self.sigmoid(W_rx + U_rh)
    z = self.sigmoid(W_zx + U_zh)
    h_tilde = self.tanh(W_hx + torch.mm(r * hidden,self.U_h_tilde))
    h = ((1-z) * hidden) + (z * h_tilde)

    return h

# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob=0.2):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(GRU, self).__init__()
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.drop_prob = 1 - dp_keep_prob

    self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
    self.decode = nn.Linear(hidden_size,vocab_size)

    self.fc = clones(nn.Linear(hidden_size, hidden_size), num_layers)
    self.dropout = clones(nn.Dropout(self.drop_prob), num_layers)
    self.softmax = nn.Softmax(dim=2)

    self.GRU_cells = nn.ModuleList([GRU_cell(emb_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    # Initialization of the parameters of the recurrent and fc layers.
    # Your implementation should support any number of stacked hidden layers
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the
    # provided clones function (as opposed to a regular python list), in order
    # for Pytorch to recognize these parameters as belonging to this nn.Module
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.

  def init_hidden(self): #called in main in the epoch loop
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # Compute the forward pass, using a nested python for loops.
    # The outer for loop should iterate over timesteps, and the
    # inner for loop should iterate over hidden layers of the stack.
    #
    # Within these for loops, use the parameter tensors and/or nn.modules you
    # created in __init__ to compute the recurrent updates according to the
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

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
    embeddings = self.embedding(inputs)
    for t in range(self.seq_len):
      h_next_ts = []
      # TODO
      # change dropout
      input = self.dropout[0](embeddings[t])
      for h_index in range(self.num_layers):
        # Recurrent GRU cell
        h_recurrent = self.GRU_cells[h_index].forward(input,h_previous_ts[h_index])
        # Fully connected layer with dropout
        h_previous_layer = self.dropout[h_index](self.fc[h_index](h_recurrent))
        input = h_previous_layer # used vertically up the layers
        # Keep the ref for next ts
        h_next_ts.append(h_recurrent) # used horizontally across timesteps
      h_previous_ts = torch.stack(h_next_ts)
      logits.append(self.decode(h_previous_layer))
    return torch.stack(logits), h_next_ts

  def generate(self, input, hidden, generated_seq_len): #generate next work using the GRU
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    #
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution,
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation
    # function here in order to compute the parameters of the categorical
    # distributions to be sampled from at each time-step.

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
    new_input = input

    for t in range(generated_seq_len):
      h_next_ts = []
      new_input = new_input.to(torch.device("cuda"))
      embedding = self.embedding(new_input)
      input = embedding
      for h_index in range(self.num_layers):
        # Recurrent GRU cell
        h_recurrent = self.GRU_cells[h_index].forward(input,h_previous_ts[h_index])
        # Fully connected layer with dropout
        h_previous_layer = self.dropout[h_index](self.fc[h_index](h_recurrent))
        input = h_previous_layer # used vertically up the layers
        # Keep the ref for next ts
        h_next_ts.append(h_recurrent) # used horizontally across timesteps

      h_previous_ts = torch.stack(h_next_ts)

      sample = h_previous_layer
      sample = self.softmax(self.decode(sample))
      sample_index = int(np.argmax(sample.cpu().detach().numpy()))
      samples.append(sample_index)
      new_input[0, 0] = sample_index

    return samples

