from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=2, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        return torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)

    def forward(self, inputs, len_seq):
        # 转置
        inputs = inputs.t()
        embedding = self.embedding(inputs)
        gru_input = pack_padded_sequence(embedding, len_seq)
        hidden = self._init_hidden(inputs.size(1)).to(inputs.device)
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output
