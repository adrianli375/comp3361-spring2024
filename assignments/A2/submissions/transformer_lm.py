# models.py

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import random
from tqdm import tqdm

from transformer import PositionalEncoding


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise NotImplementedError("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise NotImplementedError("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, num_heads, num_layers, vocab_indexer, dropout_rate: float=0.1):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; e.g. 20
        :param d_model: The dimension of the inputs and outputs of the transformer layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param num_heads: number of heads to use in multi-head attention layers; e.g. 8
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param vocab_indexer: the vocab indexer, an Indexer of the character vocabulary (27 characters)
        :param dropout_rate: the dropout rate applied to the model, default: 0.1
        """
        super(NeuralLanguageModel, self).__init__()
        # embedding, positional encoding and transformer encoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self._mask = self._generate_mask(num_positions)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, norm_first=True, batch_first=True),
            num_layers,
        )
        # layer norm
        self.layernorm = nn.LayerNorm(d_model)
        # Linear layer, nonlinearity and Linear layer
        self.linear1 = nn.Linear(d_model, d_model)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear3 = nn.Linear(d_model, d_model)
        self.activation2 = nn.ReLU()
        self.linear4 = nn.Linear(d_model, vocab_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        # vocab indexer
        self.vocab_indexer = vocab_indexer
        # other properties
        self.n = num_positions
        self.vocab_size = vocab_size
    
    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities or labels
        """
        x = self.embedding(indices)
        x = self.positional_encoding(x)
        x_size = x.shape[0]
        self._mask = self._generate_mask(x_size)
        x = x + self.transformer_encoder(x, mask=self._mask, is_causal=True)
        x = self.layernorm(x)
        x = self.linear2(self.dropout1(self.activation1(self.linear1(x))) + x)
        x = self.linear4(self.dropout2(self.activation2(self.linear3(x))) + x)
        log_probs = F.log_softmax(x, dim=-1)
        return log_probs

    def _generate_mask(self, size):
        mask = torch.triu(
            torch.full((size, size), float('-inf'), dtype=torch.float64), 
            diagonal=1)
        return mask
    
    def get_next_char_log_probs(self, context):
        self.eval()
        with torch.no_grad():
            input_indices = np.array([self.vocab_indexer.index_of(ci) for ci in context])
            input_tensor = torch.LongTensor(input_indices)
            log_probs = self.forward(input_tensor)
        return log_probs[-1, :].squeeze().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.eval()
        if context == '': # indicate the start of sequence character
            context = ' '  
        log_prob_sequence = 0.0
        for i, char in enumerate(next_chars):
            new_context = f'{context}{next_chars[:i]}'
            if len(new_context) > self.n:
                new_context = new_context[-self.n:]
            log_prob_dist = self.get_next_char_log_probs(new_context)
            char_id = self.vocab_indexer.index_of(char)
            log_prob = log_prob_dist[char_id]
            log_prob_sequence += log_prob
        return log_prob_sequence.item()


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    chunk_size = 50
    model = NeuralLanguageModel(27, chunk_size, 128, 1, 2, vocab_index)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    num_epochs = 10
    for t in range(0, num_epochs):
        random.seed(t)
        loss_this_epoch = 0.0
        ex_idxs = [i for i in range(0, len(train_text) // chunk_size)]
        loss_fcn = nn.NLLLoss()
        for ex_idx in tqdm(ex_idxs):
            optimizer.zero_grad()
            start_idx, end_idx = ex_idx * chunk_size, (ex_idx + 1) * chunk_size
            input_indices = np.array([vocab_index.index_of(ci) \
                                      for ci in f' {train_text[start_idx:end_idx-1]}'])
            inputs = torch.LongTensor(input_indices)
            label_indices = np.array([vocab_index.index_of(ci) \
                                      for ci in f'{train_text[start_idx:end_idx]}'])
            labels = torch.LongTensor(label_indices)
            # Run forward and compute loss
            outputs = model(inputs)
            loss = loss_fcn(outputs, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print(datetime.now(), f'Epoch {t+1}, loss = {loss_this_epoch}')
    model.eval()
    return model
