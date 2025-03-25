import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action encoder (learns latent action space)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Dynamics model
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, state, action):
        # Encode state
        state_encoding = self.state_encoder(state)
        
        # Learn latent action representation
        action_latent = self.action_encoder(action)
        
        # Predict next state
        combined = torch.cat([state_encoding, action_latent], dim=-1)
        next_state = self.dynamics(combined)
        
        return next_state, action_latent
    
    def update(self, state, action, next_state):
        """Update the world model using experience"""
        # Forward pass
        predicted_next_state, action_latent = self(state, action)
        
        # Compute losses
        prediction_loss = nn.MSELoss()(predicted_next_state, next_state)
        latent_regularization = 0.01 * torch.mean(torch.abs(action_latent))
        total_loss = prediction_loss + latent_regularization
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'prediction_loss': prediction_loss.item(),
            'latent_reg': latent_regularization.item(),
            'total_loss': total_loss.item()
        }