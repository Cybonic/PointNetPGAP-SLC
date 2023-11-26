import torch
import torch.nn as nn
import torch.nn.functional as F
from multihead import GeM, SPoC, MAC
class Expert(nn.Module):
    def __init__(self, input_size, output_size):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.relu(self.fc(x))

class MoE(nn.Module):
    def __init__(self, input_size, expert_output_size, output_size):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([  SPoC(outdim=expert_output_size),
                                        GeM(outdim=expert_output_size),
                                        MAC(outdim=expert_output_size)])
        self.gating_network = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Softmax(dim=1)
        )
        self.final_layer = nn.Linear(3, output_size)

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: (batch_size, num_experts, expert_output_size)

        gate_weights = self.gating_network(x)  # Shape: (batch_size, num_experts)

        # Weighted sum of expert outputs based on gating weights
        weighted_sum = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)

        # Final output
        output = self.final_layer(weighted_sum)
        return output

if __name__=="__main__":
    # Example usage:
    input_size = 1024
    expert_output_size = 256
    num_experts = 3
    output_size = 256

    model = MoE(input_size, expert_output_size, output_size)

    # Example input tensor
    x = torch.randn(2,1024,10000)


    # Forward pass
    output = model(x)
    print(output.shape)  # Check the output shape
