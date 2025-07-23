"""
Deep Deterministic Policy Gradient (DDPG) Agent
Implements DDPG algorithm for continuous control in NDN network congestion management
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Optional
import copy
import logging

class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration in continuous action spaces"""
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        """Reset the internal state to mean"""
        self.state = copy.copy(self.mu)
        
    def sample(self) -> np.ndarray:
        """Update internal state and return noise sample"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

class ReplayBuffer:
    """Experience replay buffer for DDPG training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done).unsqueeze(1)
        )
        
    def __len__(self) -> int:
        return len(self.buffer)

class Actor(nn.Module):
    """Actor network for DDPG - outputs continuous actions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
            
        # Final layer with small weights for stable initial policy
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias, -3e-3, 3e-3)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network"""
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        action = torch.tanh(self.fc4(x))  # Action in [-1, 1]
        return action

class Critic(nn.Module):
    """Critic network for DDPG - estimates Q-values"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # State processing layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Combined state-action processing
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # Layer normalization
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
            
        # Final layer with small weights
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias, -3e-3, 3e-3)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network"""
        # Process state
        x = F.relu(self.ln1(self.fc1(state)))
        
        # Combine state and action
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        q_value = self.fc4(x)
        
        return q_value

class DDPGAgent:
    """Deep Deterministic Policy Gradient Agent for NDN congestion control"""
    
    def __init__(self, config: Dict):
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dim = config.get('hidden_dim', 256)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)  # Soft update parameter
        self.memory_size = config.get('memory_size', 100000)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        
        # Initialize target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size)
        
        # Noise for exploration
        self.noise = OUNoise(self.action_dim)
        
        # Performance tracking
        self.training_step = 0
        self.episode_rewards = []
        self.losses = {'actor': [], 'critic': []}
        
        # NDN-specific state tracking
        self.congestion_history = deque(maxlen=100)
        self.action_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        self.logger = logging.getLogger(f"DDPGAgent")
        
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        
        # Add noise for exploration
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Track action for analysis
        self.action_history.append(action.copy())
        
        return action
        
    def step(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store experience and potentially train"""
        # Store experience
        self.memory.push(state, action, reward, next_state, done)
        
        # Track reward
        self.reward_history.append(reward)
        
        # Train if enough experiences
        if len(self.memory) > self.batch_size:
            self.train()
            
    def train(self):
        """Train actor and critic networks"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Train Critic
        self.critic_optimizer.zero_grad()
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.gamma * target_q * ~dones)
            
        # Compute current Q-values
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        critic_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Train Actor
        self.actor_optimizer.zero_grad()
        
        # Actor loss (negative because we want to maximize Q-value)
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        # Add regularization term to prevent large actions
        action_penalty = 0.01 * (actor_actions ** 2).mean()
        actor_loss += action_penalty
        
        actor_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        # Track losses
        self.losses['actor'].append(actor_loss.item())
        self.losses['critic'].append(critic_loss.item())
        
        self.training_step += 1
        
    def soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update of target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
    def hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update of target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, 
                        next_state: np.ndarray, network_metrics: Dict) -> float:
        """Calculate reward based on NDN network performance"""
        # Network performance components
        congestion_level = next_state[0]  # Congestion level
        latency = next_state[4]  # Normalized latency
        packet_loss = next_state[3]  # Packet loss rate
        cache_hit_rate = next_state[2]  # Cache hit rate
        throughput = network_metrics.get('throughput', 10.0) / 20.0  # Normalized
        
        # Reward components
        congestion_penalty = -10.0 * congestion_level ** 2
        latency_penalty = -5.0 * latency ** 2
        loss_penalty = -15.0 * packet_loss ** 2
        cache_reward = 3.0 * cache_hit_rate
        throughput_reward = 2.0 * throughput
        
        # Action penalty (discourage large actions)
        action_penalty = -0.1 * np.sum(action ** 2)
        
        # Stability bonus (reward smooth actions)
        if len(self.action_history) > 1:
            action_change = np.linalg.norm(action - self.action_history[-2])
            stability_bonus = -0.5 * action_change
        else:
            stability_bonus = 0.0
            
        # Federated learning bonus (if applicable)
        fl_bonus = 0.0
        if hasattr(self, 'is_federated') and self.is_federated:
            fl_bonus = 1.0  # Bonus for participating in federated learning
            
        total_reward = (
            congestion_penalty + latency_penalty + loss_penalty +
            cache_reward + throughput_reward + action_penalty +
            stability_bonus + fl_bonus
        )
        
        # Track congestion for analysis
        self.congestion_history.append(congestion_level)
        
        return np.clip(total_reward, -50.0, 50.0)  # Clip reward for stability
        
    def get_model_parameters(self) -> Dict:
        """Get model parameters for federated learning"""
        return {
            'actor': {
                name: param.cpu().data.clone() 
                for name, param in self.actor.named_parameters()
            },
            'critic': {
                name: param.cpu().data.clone() 
                for name, param in self.critic.named_parameters()
            }
        }
        
    def set_model_parameters(self, parameters: Dict):
        """Set model parameters from federated learning"""
        # Update actor parameters
        if 'actor' in parameters:
            actor_dict = self.actor.state_dict()
            for name, param in parameters['actor'].items():
                if name in actor_dict:
                    actor_dict[name].copy_(param.to(self.device))
            self.actor.load_state_dict(actor_dict)
            
        # Update critic parameters
        if 'critic' in parameters:
            critic_dict = self.critic.state_dict()
            for name, param in parameters['critic'].items():
                if name in critic_dict:
                    critic_dict[name].copy_(param.to(self.device))
            self.critic.load_state_dict(critic_dict)
            
        # Update target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
    def get_performance_summary(self) -> Dict:
        """Get performance summary for federated learning quality assessment"""
        recent_rewards = list(self.reward_history)[-50:] if self.reward_history else [0]
        recent_actions = list(self.action_history)[-50:] if self.action_history else [np.zeros(self.action_dim)]
        recent_congestion = list(self.congestion_history)[-50:] if self.congestion_history else [0.5]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'reward_std': np.std(recent_rewards),
            'avg_critic_loss': np.mean(self.losses['critic'][-50:]) if self.losses['critic'] else 1.0,
            'avg_actor_loss': np.mean(self.losses['actor'][-50:]) if self.losses['actor'] else 1.0,
            'action_variance': np.var(recent_actions, axis=0).mean() if len(recent_actions) > 1 else 0.5,
            'congestion_control': 1.0 - np.mean(recent_congestion),
            'replay_buffer_size': len(self.memory),
            'training_steps': self.training_step
        }
        
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'tau': self.tau
            }
        }
        torch.save(checkpoint, filepath)
        
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        
        # Update target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
    def reset_episode(self):
        """Reset for new episode"""
        self.noise.reset()
        
    def reset(self):
        """Reset agent completely"""
        # Clear memory
        self.memory = ReplayBuffer(self.memory_size)
        
        # Reset noise
        self.noise.reset()
        
        # Clear tracking variables
        self.training_step = 0
        self.episode_rewards = []
        self.losses = {'actor': [], 'critic': []}
        self.congestion_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        
        # Reinitialize networks
        self.actor.apply(self._init_weights)
        self.critic.apply(self._init_weights)
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        self.logger.info("DDPG agent reset completed")
        
    def _init_weights(self, layer):
        """Initialize weights for network layers"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
