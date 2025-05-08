# module 5a: Define the Model Architecture - ctrnn_model_checkpoint.pth
# version: 1.0
# function: Define the Model Architecture

import torch
import torch.nn as nn
from torch.optim import Adam
from torchdiffeq import odeint


# Define the Neural ODE block that evolves with external inputs
class BatchedDynamicsFunctionWithFusedInput(nn.Module):
    def __init__(self, hidden_dim):
        super(BatchedDynamicsFunctionWithFusedInput, self).__init__()
        self.linear = nn.Linear(hidden_dim + 1, hidden_dim)  # +1 for time dimension
        self.activation = nn.Tanh()

    def forward(self, t, state):
        # Create the time tensor and broadcast it to match the batch size
        t_tensor = t.view(1, 1).expand(state.size(0), 1).to(state.device)

        # Concatenate state and time along the feature dimension
        combined_input = torch.cat([state, t_tensor], dim=-1)  # Shape: [batch_size, hidden_dim + 1]
        state_derivative = self.activation(self.linear(combined_input))
        return state_derivative

# Continuous Time RNN (CT-RNN) model with classification head
class CTRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CTRNN, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)  # Linear layer to transform input to hidden_dim
        self.dynamics_function = BatchedDynamicsFunctionWithFusedInput(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, initial_state, t):
        # Use a linear layer to transform the input state to hidden_dim size
        initial_state_transformed = torch.tanh(self.input_to_hidden(initial_state))
        
        # Pass the transformed initial state to odeint
        solution = odeint(self.dynamics_function, initial_state_transformed, t, method='dopri5')
        last_hidden_state = solution[-1]  # Use the last hidden state for classification
        prediction = self.classifier(last_hidden_state)
        return prediction