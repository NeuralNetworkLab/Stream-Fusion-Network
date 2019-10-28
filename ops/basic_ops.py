import torch
import math
import torch.nn as nn

class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None


    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        if self.consensus_type == 'lstm':
            self.class_num = 101
            self.lstm_fuse = nn.LSTM(self.class_num, self.class_num, 2, batch_first=True)

    def forward(self, input):
        if self.consensus_type=='lstm':
            # inputs = input_tensor.view(self.shape[self.dim], -1, self.class_num)                    # segments number==3 ,batch_size, self.class_num=101
            hidden = (torch.zeros(2, input.shape[0], self.class_num).cuda(), torch.zeros(2, input.shape[0], self.class_num).cuda())  # clean out hidden state
            out, hidden = self.lstm_fuse(input,hidden)
            output=out[:,-1,:]
            return output
        else:
            return SegmentConsensus(self.consensus_type, self.dim)(input)
