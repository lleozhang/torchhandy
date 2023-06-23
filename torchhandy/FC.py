import torch.nn as nn
import torch

def select_activation(activation):
    act = None
    if activation == 'relu':
        act = nn.ReLU()
    elif activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'softmax':
        act = nn.Softmax()
    elif activation == 'elu':
        act = nn.ELU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation == 'gelu':
        act = nn.GELU()
        
    return act


def select_normalization(normalization):
    norm = None
    try:
        if normalization[0] == 1:
            norm = nn.BatchNorm1d(normalization[1])
        elif normalization[0] == 2:
            norm = nn.BatchNorm2d(normalization[1])
        elif normalization[0] == 0:
            norm = nn.LayerNorm(normalization[1])    
    except Exception as e:
        raise Warning("No or unsupported normalization type given!")

    return norm

'''
    This is a module of a fully connected layer with optional dropout, bias, normalization, 
    activation and residual connection.
'''
class FC(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias = True, 
                normalization = None, activation = 'relu', res_connection = False):
        '''
            input_dim: The dimension of the input, which is the size of the last dimension
                        of the input.
            output_dim: The dimension of the output, which is the size of the last dimension
                        of the output.
            dropout: The ratio of dropout. 0 means no dropout. 
            bias: Whether bias is used in Linear Layer.
            normalization: A tuple indicating which kind of normalization is used, 
            3 types are supported: (0, size) indicates layer normalization with size, 
                                    (1, channels) indicates 1d batch normalization on channels,
                                    (2, channels) indicates 2d batch normalization on channels.
            activation: A string indicating which kind of activation function to use. 
                        All letters should be small.
            res_connection: A bool indicating whether residual connection is used, only possible 
                            if input dim equals to output_dim.
        '''
        
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_connection = res_connection
                
        self.li = nn.Linear(input_dim, output_dim, bias = bias)
        self.norm = select_normalization(normalization)
        self.act = select_activation(activation)        
        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        output = self.li(X)
        if self.norm:
            output = self.norm(output)
        if self.input_dim == self.output_dim and self.res_connection:
            output += X
        if self.act:
            output = self.act(output)
        output = self.drop(output)
        return output
