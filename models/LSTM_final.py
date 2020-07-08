import torch
import torch.nn as nn
from opts import *

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed)

class LSTM_final(nn.Module):
    def __init__(self):
        super(LSTM_final, self).__init__()
        self.rnn = nn.LSTM(features_dim, 256, 1, batch_first=True)
        self.fc = nn.Linear(256,1)
        self.act = nn.Sigmoid() # nn.ReLU()

    def forward(self, x):
        state = None
        lstm_output, state = self.rnn(x, state)
        final_skill_score = self.act(self.fc(lstm_output[:,-1,:]))
        return final_skill_score, lstm_output[:,-1,:]