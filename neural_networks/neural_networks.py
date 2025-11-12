"""
Neural Network Architectures for Battlesnake AI

This module contains PyTorch neural network architectures for:
1. Position Evaluation Network - Evaluates board positions for strategic value
2. Move Prediction Network - Predicts move probability distributions
3. Game Outcome Network - Predicts win probability from board state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out += identity  # Skip connection
        return F.relu(out)

class PositionEvaluationNetwork(nn.Module):
    """
    Position Evaluation Network
    
    Input: Board state (batch_size, channels, height, width) + features (batch_size, feature_dim)
    Output: Position quality score (batch_size, 1)
    
    Architecture:
    - CNN backbone for spatial feature extraction
    - Global average pooling
    - Feature concatenation and MLP
    - Final position score output
    """
    
    def __init__(self, 
                 num_channels: int = 7,
                 max_board_size: int = 20,
                 feature_dim: int = 6,
                 conv_filters: List[int] = [32, 64, 128],
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_channels = num_channels
        self.max_board_size = max_board_size
        self.feature_dim = feature_dim
        
        # CNN backbone
        self.conv_layers = nn.ModuleList()
        
        # First conv layer
        self.conv_layers.append(ConvBlock(num_channels, conv_filters[0]))
        
        # Residual blocks
        for i in range(1, len(conv_filters)):
            self.conv_layers.append(ResidualBlock(conv_filters[i-1]))
            self.conv_layers.append(ConvBlock(conv_filters[i-1], conv_filters[i]))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature processing MLP
        conv_output_dim = conv_filters[-1]
        
        # Combine CNN features with board features
        total_feature_dim = conv_output_dim + feature_dim
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, grid: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch_size = grid.size(0)
        
        # CNN feature extraction
        x = grid
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, conv_filters[-1], 1, 1)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, conv_filters[-1])
        
        # Concatenate with hand-crafted features
        x = torch.cat([x, features], dim=1)
        
        # Final position evaluation
        position_score = self.feature_mlp(x)
        
        return position_score
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

class MovePredictionNetwork(nn.Module):
    """
    Move Prediction Network
    
    Input: Board state (batch_size, channels, height, width) + features (batch_size, feature_dim)
    Output: Move probabilities (batch_size, 4) for [up, down, left, right]
    
    Architecture:
    - CNN backbone for spatial feature extraction
    - Global average pooling
    - Feature concatenation and MLP
    - Softmax output for move probabilities
    """
    
    def __init__(self,
                 num_channels: int = 7,
                 max_board_size: int = 20,
                 feature_dim: int = 6,
                 conv_filters: List[int] = [32, 64, 128],
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_channels = num_channels
        self.max_board_size = max_board_size
        self.feature_dim = feature_dim
        
        # CNN backbone
        self.conv_layers = nn.ModuleList()
        
        # First conv layer
        self.conv_layers.append(ConvBlock(num_channels, conv_filters[0]))
        
        # Residual blocks
        for i in range(1, len(conv_filters)):
            self.conv_layers.append(ResidualBlock(conv_filters[i-1]))
            self.conv_layers.append(ConvBlock(conv_filters[i-1], conv_filters[i]))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature processing MLP
        conv_output_dim = conv_filters[-1]
        total_feature_dim = conv_output_dim + feature_dim
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),  # 4 moves
            nn.Softmax(dim=1)  # Ensure probabilities sum to 1
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, grid: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch_size = grid.size(0)
        
        # CNN feature extraction
        x = grid
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, conv_filters[-1], 1, 1)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, conv_filters[-1])
        
        # Concatenate with hand-crafted features
        x = torch.cat([x, features], dim=1)
        
        # Final move prediction
        move_probs = self.feature_mlp(x)
        
        return move_probs
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

class GameOutcomeNetwork(nn.Module):
    """
    Game Outcome Network
    
    Input: Board state (batch_size, channels, height, width) + features (batch_size, feature_dim)
    Output: Win probability (batch_size, 1)
    
    Architecture:
    - CNN backbone for spatial feature extraction
    - Global average pooling
    - Feature concatenation and MLP
    - Sigmoid output for win probability
    """
    
    def __init__(self,
                 num_channels: int = 7,
                 max_board_size: int = 20,
                 feature_dim: int = 6,
                 conv_filters: List[int] = [32, 64, 128, 256],
                 hidden_dim: int = 512):
        super().__init__()
        
        self.num_channels = num_channels
        self.max_board_size = max_board_size
        self.feature_dim = feature_dim
        
        # CNN backbone (deeper for game outcome prediction)
        self.conv_layers = nn.ModuleList()
        
        # First conv layer
        self.conv_layers.append(ConvBlock(num_channels, conv_filters[0]))
        
        # Residual blocks with increasing filters
        for i in range(1, len(conv_filters)):
            # Add residual block
            self.conv_layers.append(ResidualBlock(conv_filters[i-1]))
            
            # Add conv layer to change channels
            if i < len(conv_filters):
                self.conv_layers.append(ConvBlock(conv_filters[i-1], conv_filters[i]))
        
        # Additional residual layers for deeper feature extraction
        for _ in range(2):
            self.conv_layers.append(ResidualBlock(conv_filters[-1]))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature processing MLP (larger for complex game outcome)
        conv_output_dim = conv_filters[-1]
        total_feature_dim = conv_output_dim + feature_dim
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Win probability
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, grid: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch_size = grid.size(0)
        
        # CNN feature extraction
        x = grid
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, conv_filters[-1], 1, 1)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, conv_filters[-1])
        
        # Concatenate with hand-crafted features
        x = torch.cat([x, features], dim=1)
        
        # Final win probability
        win_prob = self.feature_mlp(x)
        
        return win_prob
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

# Network factory functions
def create_position_network(num_channels: int = 7, 
                          max_board_size: int = 20,
                          feature_dim: int = 6,
                          conv_filters: List[int] = None,
                          hidden_dim: int = 256) -> PositionEvaluationNetwork:
    """Create position evaluation network with defaults"""
    if conv_filters is None:
        conv_filters = [32, 64, 128]
    return PositionEvaluationNetwork(num_channels, max_board_size, feature_dim, conv_filters, hidden_dim)

def create_move_network(num_channels: int = 7,
                      max_board_size: int = 20,
                      feature_dim: int = 6,
                      conv_filters: List[int] = None,
                      hidden_dim: int = 256) -> MovePredictionNetwork:
    """Create move prediction network with defaults"""
    if conv_filters is None:
        conv_filters = [32, 64, 128]
    return MovePredictionNetwork(num_channels, max_board_size, feature_dim, conv_filters, hidden_dim)

def create_outcome_network(num_channels: int = 7,
                         max_board_size: int = 20,
                         feature_dim: int = 6,
                         conv_filters: List[int] = None,
                         hidden_dim: int = 512) -> GameOutcomeNetwork:
    """Create game outcome network with defaults"""
    if conv_filters is None:
        conv_filters = [32, 64, 128, 256]
    return GameOutcomeNetwork(num_channels, max_board_size, feature_dim, conv_filters, hidden_dim)

def test_networks():
    """Test network architectures with dummy data"""
    batch_size = 4
    num_channels = 7
    max_board_size = 20
    feature_dim = 6
    
    # Create dummy input
    grid = torch.randn(batch_size, num_channels, max_board_size, max_board_size)
    features = torch.randn(batch_size, feature_dim)
    
    print("Testing Position Evaluation Network...")
    pos_net = create_position_network()
    pos_output = pos_net(grid, features)
    print(f"Position output shape: {pos_output.shape}")
    print(f"Position output range: [{pos_output.min().item():.3f}, {pos_output.max().item():.3f}]")
    
    print("\nTesting Move Prediction Network...")
    move_net = create_move_network()
    move_output = move_net(grid, features)
    print(f"Move output shape: {move_output.shape}")
    print(f"Move probabilities sum: {move_output.sum(dim=1)}")
    
    print("\nTesting Game Outcome Network...")
    outcome_net = create_outcome_network()
    outcome_output = outcome_net(grid, features)
    print(f"Outcome output shape: {outcome_output.shape}")
    print(f"Outcome output range: [{outcome_output.min().item():.3f}, {outcome_output.max().item():.3f}]")
    
    # Count parameters
    pos_params = sum(p.numel() for p in pos_net.parameters())
    move_params = sum(p.numel() for p in move_net.parameters())
    outcome_params = sum(p.numel() for p in outcome_net.parameters())
    
    print(f"\nParameter counts:")
    print(f"Position Network: {pos_params:,}")
    print(f"Move Network: {move_params:,}")
    print(f"Outcome Network: {outcome_params:,}")

if __name__ == "__main__":
    test_networks()