import torch
import torch.nn as nn

class FuzzyLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FuzzyLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        mean_value_weights = torch.Tensor(1, self.output_dim*self.input_dim)
        self.mean_value = nn.Parameter(mean_value_weights)

        sigma_weights = torch.Tensor(1, self.output_dim*self.input_dim)
        self.sigma = nn.Parameter(sigma_weights)

        # initialize fuzzy degree and sigma parameters
        nn.init.xavier_uniform_(self.mean_value)  
        nn.init.ones_(self.sigma) 

    def forward(self, input):
        # fuzzy membership layer
        input_expanded = input.repeat(1, self.output_dim)
        fuzz_membership = (torch.exp(-(input_expanded - self.mean_value).pow(2) / (self.sigma.pow(2))))
       
        # fuzzy rule layer
        fuzz_output = fuzz_membership.view(fuzz_membership.shape[0], self.input_dim, self.output_dim).prod(dim=1)

        return fuzz_output
    