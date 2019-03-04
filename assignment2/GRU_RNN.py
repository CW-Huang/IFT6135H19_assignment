import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# TODO: Generalize to multi-layers

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

    self.W_x = nn.Parameter(torch.FloatTensor(emb_size,3*hidden_size)) #.cuda()
    self.U_h = nn.Parameter(torch.FloatTensor(hidden_size,2*hidden_size))
    self.U_h_tilde = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size))
    self.bias_rzh = nn.Parameter(torch.FloatTensor(3*hidden_size))

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

    # add batch_size as a dimension to bias since I will be adding the bias below (need the dimensions to match)
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
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
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
    self.dp_keep_prob = dp_keep_prob
    self.decoder = nn.Linear(hidden_size,vocab_size)

    # The size of the input to the first GRU cell is of size embedding size.
    # The size of the inputs to all other GRU cells is of size hidden size.
    # There's one GRU cell per hidden layer
    self.GRU_cells = [GRU_cell(emb_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
    self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

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
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """

    return # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # TODO ========================
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
    embeddings = self.embedding(inputs) #(seq_len, batch_size, emb_size)
    logits = []
    for t in range(len(self.seq_len)):
      input = embeddings[t] #the very first input to a GRU cell is the embedding (#embeddings == #seq_len)
      for h_index in range(self.num_layers):
        h = self.GRU_cells[h_index].forward(input,hidden[h_index])
        input = h # updating the input with output of the GRU cell, will be used as input to GRU cell at next timesept (vertically up the stacks)
        hidden[h_index]=h # updating the hidden layer with output of GRU cell, will be used as input to GRU cell at next timestep (horizontally)
        #hidden state of the last cell is assecbile here
      logits.append(self.decode(h))

    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len): #generate next work using the GRU
    # TODO ========================
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
    return #samples

#used temporarily for testing
if __name__ == '__main__':
  gru_cell = GRU_cell(20,30) #(emb,hidden_size)
  inp = torch.rand(5,20) #(batch_size, emb)
  hid = torch.zeros(5,30) #(batch_size, hidden_size)

  results = gru_cell.forward(inp,hid)
  print(results)

