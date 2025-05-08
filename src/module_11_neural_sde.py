"""
module_11_neural_sde.py
-----------------------------------------
Defines the NeuralSDE class which parameterizes drift (mu) and diffusion (sigma) via MLPs,
and an SDEFunction class (with noise_type, sde_type) for torchsde.sdeint().
"""

import torch
import torch.nn as nn
import torchsde

class NeuralSDE(nn.Module):
    """
    Parameterizes drift (mu) and diffusion (sigma) via MLPs.
    
    h_dim: dimension of the hidden state (embedding dimension in your pipeline).
    param_dim_f: hidden dimension for the drift network.
    param_dim_g: hidden dimension for the diffusion network.
    """
    def __init__(self, h_dim: int, param_dim_f: int, param_dim_g: int):
        super(NeuralSDE, self).__init__()
        # Drift net
        self.drift_net = nn.Sequential(
            nn.Linear(h_dim, param_dim_f),
            nn.ReLU(),
            nn.Linear(param_dim_f, h_dim)
        )
        # Diffusion net
        self.diffusion_net = nn.Sequential(
            nn.Linear(h_dim, param_dim_g),
            nn.ReLU(),
            nn.Linear(param_dim_g, h_dim)
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """
        :param t: Current time, shape [] or [batch_size], but not typically used here.
        :param x: State, shape [batch_size, h_dim].
        :return: (drift, diffusion) both shape [batch_size, h_dim].
        """
        mu = self.drift_net(x)      # drift
        sigma = self.diffusion_net(x)  # diffusion
        return mu, sigma


class SDEFunction(nn.Module):
    """
    A wrapper for torchsde.sdeint to integrate the SDE.
    Must define:
      - sde_type: "ito" or "stratonovich"
      - noise_type: "diagonal", "additive", "general", etc.
      - f(t, x): drift
      - g(t, x): diffusion
    """

    # REQUIRED attributes for torchsde.sdeint():
    sde_type = 'ito'       # or 'stratonovich'
    noise_type = 'diagonal'  # or 'general', 'scalar', 'additive', etc.

    def __init__(self, neural_sde: NeuralSDE):
        super(SDEFunction, self).__init__()
        self.neural_sde = neural_sde

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Drift function. (a.k.a. 'f' in SDEs, meaning dX_t = f(t, X_t) dt + g(t, X_t) dW_t)
        :param t: shape []
        :param x: [batch_size, h_dim]
        :return: drift, same shape [batch_size, h_dim]
        """
        mu, _ = self.neural_sde(t, x)
        return mu

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion function. Called 'g' in SDEs.
        We produce a diagonal matrix from sigma so that noise is dimension h_dim.
        
        :param t: shape []
        :param x: [batch_size, h_dim]
        :return: shape [batch_size, h_dim, h_dim] if noise_type='diagonal'
                 or [batch_size, h_dim, 1] if noise_type='scalar'
        """
        _, sigma = self.neural_sde(t, x)

        # For a diagonal diffusion, we do diag_embed(sigma).
        # sigma shape: [batch_size, h_dim] => returns [batch_size, h_dim, h_dim]
        # return torch.diag_embed(sigma)
        return sigma
