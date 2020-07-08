import torch
import torch.nn as nn
import numpy as np
import random
from opts import*

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)

class skill_classifier(nn.Module):
    def __init__(self):
        super(skill_classifier, self).__init__()
        self.fc_gesture_impression = nn.Linear(256, num_gesture_types)
        self.fc_skill_class = nn.Linear(256, 3)
        self.act = nn.Softmax()

        # self.svr = nn.Linear(lstm_indim, 1)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        ges_imp = self.fc_gesture_impression(x)
        sp_skill = self.act(self.fc_skill_class(x))
        if mode == 'MTL-VF':
            return sp_skill
        elif mode == 'IMTL-AGF':
            return ges_imp

        # return self.act(self.svr(x))