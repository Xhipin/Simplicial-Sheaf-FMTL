import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

class SheafFMTL:
    """
    Sheaf-based Federated Multi-Task Learning Algorithm
    
    Args:
        models: List of client models
        graph: NetworkX graph representing client connectivity
        lambda_reg: Regularization parameter
        alpha: Learning rate for model parameters
        eta: Learning rate for restriction maps
        gamma: Factor controlling dimension of interaction space
    """
    
    def __init__(self, models: List[nn.Module], graph, 
                 lambda_reg: float = 0.1, 
                 alpha: float = 0.001,
                 eta: float = 0.001,
                 gamma: float = 0.1):
        self.models = models
        self.graph = graph
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma
        self.num_clients = len(models)
        
        # Initialize restriction maps
        self.P = self._initialize_restriction_maps()
        
    def _initialize_restriction_maps(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Initialize restriction maps P_ij for all edges in the graph"""
        P = {}
        
        for i, j in self.graph.edges():
            # Calculate dimensions
            num_params_i = sum(p.numel() for p in self.models[i].parameters())
            num_params_j = sum(p.numel() for p in self.models[j].parameters())
            
            # Dimension of interaction space
            d_ij = int(self.gamma * min(num_params_i, num_params_j))
            
            # Initialize restriction maps
            P[(i, j)] = torch.randn(d_ij, num_params_i) * 0.01
            P[(j, i)] = torch.randn(d_ij, num_params_j) * 0.01
            
        return P
    
    def local_update(self, client_id: int, dataloader, 
                    local_epochs: int = 1, l2_strength: float = 0.01):
        """Perform local training on client's data"""
        model = self.models[client_id]
        optimizer = torch.optim.SGD(model.parameters(), lr=self.alpha)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(local_epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Add L2 regularization
                l2_reg = sum(param.pow(2).sum() for param in model.parameters())
                loss = loss + l2_strength * l2_reg
                
                loss.backward()
                optimizer.step()
    
    def sheaf_update(self, client_id: int):
        """Update client model using sheaf Laplacian regularization"""
        with torch.no_grad():
            # Extract theta_i as a vector
            theta_i = self._get_model_params(client_id)
            
            # Compute sheaf Laplacian term
            sum_P_terms = torch.zeros_like(theta_i)
            
            for j in self.graph.neighbors(client_id):
                P_ij = self.P[(client_id, j)]
                P_ji = self.P[(j, client_id)]
                theta_j = self._get_model_params(j)
                
                # Compute P_ij @ theta_i - P_ji @ theta_j
                sum_P_terms += P_ij.T @ (P_ij @ theta_i - P_ji @ theta_j)
            
            # Update theta_i
            theta_i -= self.alpha * self.lambda_reg * sum_P_terms
            
            # Put updated parameters back into model
            self._set_model_params(client_id, theta_i)
    
    def update_restriction_maps(self, client_id: int):
        """Update restriction maps P_ij"""
        with torch.no_grad():
            theta_i = self._get_model_params(client_id)
            
            for j in self.graph.neighbors(client_id):
                P_ij = self.P[(client_id, j)]
                P_ji = self.P[(j, client_id)]
                theta_j = self._get_model_params(j)
                
                # Update P_ij
                diff = P_ij @ theta_i - P_ji @ theta_j
                self.P[(client_id, j)] -= self.eta * self.lambda_reg * torch.outer(diff, theta_i)
    
    def _get_model_params(self, client_id: int) -> torch.Tensor:
        """Extract model parameters as a flattened vector"""
        return torch.cat([param.view(-1) for param in self.models[client_id].parameters()])
    
    def _set_model_params(self, client_id: int, params: torch.Tensor):
        """Set model parameters from a flattened vector"""
        idx = 0
        for param in self.models[client_id].parameters():
            numel = param.numel()
            param.data.copy_(params[idx:idx+numel].reshape(param.size()))
            idx += numel
    
    def train(self, client_dataloaders: List, num_rounds: int = 100, 
              local_epochs: int = 1, evaluate_fn: Optional = None):
        """Main training loop"""
        history = {'accuracy': [], 'loss': [], 'communication_bits': []}
        cumulative_bits = 0
        
        for round_idx in range(num_rounds):
            # Local updates and sheaf updates for all clients
            for client_id in range(self.num_clients):
                # Local training
                self.local_update(client_id, client_dataloaders[client_id], 
                                local_epochs=local_epochs)
                
                # Sheaf update
                self.sheaf_update(client_id)
                
                # Update restriction maps
                self.update_restriction_maps(client_id)
            
            # Calculate communication cost
            round_bits = self._calculate_communication_bits()
            cumulative_bits += round_bits
            
            # Evaluate if function provided
            if evaluate_fn is not None:
                accuracy = evaluate_fn(self.models)
                history['accuracy'].append(accuracy)
                history['communication_bits'].append(cumulative_bits)
                
                if round_idx % 10 == 0:
                    print(f"Round {round_idx}: Avg Accuracy = {accuracy:.4f}")
        
        return history
    
    def _calculate_communication_bits(self) -> int:
        """Calculate communication bits for one round"""
        bits = 0
        bits_per_param = 32  # Assuming 32-bit floats
        
        for i in range(self.num_clients):
            for j in self.graph.neighbors(i):
                d_ij = self.P[(i, j)].shape[0]
                bits += 2 * d_ij * bits_per_param  # Two-way communication
        
        return bits
