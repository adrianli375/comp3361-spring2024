# transformer.py

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, add_positional_encoding: bool = False):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param add_positional_encoding: adds a PositionalEncoding layer if set to True.  
        """
        super().__init__()
        self.n = num_positions
        self.vocab_size = vocab_size
        self.layers = nn.ModuleList()
        # (3) Using linear layers for input
        self.layers.append(nn.Linear(vocab_size, d_model))
        # (1) add a PositionalEmbedding layer
        if add_positional_encoding:
            self.layers.append(PositionalEncoding(d_model))
        for _ in range(num_layers):
            # (2) Using transformer layers
            self.layers.append(TransformerLayer(d_model, d_internal))
        # (3) Using linear layers for output
        self.layers.append(nn.Linear(d_model, num_classes))

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        x = self.embedding(indices)
        for layer in self.layers:
            x = layer(x)
        log_probs = torch.log(F.softmax(x, dim=-1))
        # get the attentions
        attentions_list = []
        for layer in self.layers:
            if isinstance(layer, TransformerLayer):
                if layer.attentions is not None:
                    attentions_list.append(layer.attentions)
        return (log_probs, attentions_list)

    def embedding(self, indices: torch.Tensor) -> torch.Tensor:
        x = torch.zeros((self.n, self.vocab_size), dtype=torch.float)
        x.scatter_(1, indices.unsqueeze(1), 1)
        return x


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # (1) Query, Key and Value matrices
        self.W_Q = nn.Linear(d_model, d_internal)
        self.W_K = nn.Linear(d_model, d_internal)
        self.W_V = nn.Linear(d_model, d_model)
        # scaling factor used to stabilize training
        self.scaling_factor = 1 / np.sqrt(d_internal)

        # (3) Linear layer, nonlinearity and Linear layer
        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(d_model, d_model)

        # attentions computed
        self.attentions = None

    def forward(self, input_vecs):
        Q = self.W_Q(input_vecs)
        K = self.W_K(input_vecs)
        V = self.W_V(input_vecs)
        A = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling_factor
        self.attentions = F.softmax(A, dim=-1)
        attention_output = torch.matmul(self.attentions, V)

        # (2) Residual connection from the input to the attention output
        attention_output += input_vecs

        # transform the attention output to linear output in (3)
        linear_output = self.linear2(self.activation(self.linear1(attention_output)))

        # (4) Final residual connection
        output = linear_output + attention_output
        return output


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# code to train classifier
def train_classifier(args, train, dev):
    count_only_previous = True if args.task == "BEFORE" else False
    model = Transformer(27, 20, 72, 96, 3, 2, add_positional_encoding=count_only_previous)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in tqdm(ex_idxs):
            optimizer.zero_grad()
            inputs = train[ex_idx].input_tensor
            labels = train[ex_idx].output_tensor
            # Run forward and compute loss
            outputs = model(inputs)[0]
            loss = loss_fcn(outputs, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print(datetime.now(), f'Epoch {t+1}, loss = {loss_this_epoch}')
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
