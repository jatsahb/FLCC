"""
Differential Privacy Manager for Federated Learning
Implements differential privacy mechanisms to protect participant data in federated NDN learning
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import math
from collections import defaultdict

class DifferentialPrivacyManager:
    """Manages differential privacy for federated learning in NDN networks"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 noise_multiplier: float = 1.1, max_grad_norm: float = 1.0):
        """
        Initialize differential privacy manager
        
        Args:
            epsilon: Privacy budget parameter (smaller = more private)
            delta: Probability of privacy failure
            noise_multiplier: Multiplier for noise scale
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        # Privacy accounting
        self.privacy_spent = 0.0
        self.composition_steps = 0
        self.privacy_history = []
        
        # Noise calibration
        self.sensitivity = self._calculate_sensitivity()
        self.noise_scale = self._calculate_noise_scale()
        
        self.logger = logging.getLogger("DifferentialPrivacyManager")
        self.logger.info(f"Initialized DP manager: ε={epsilon}, δ={delta}, noise_mult={noise_multiplier}")
        
    def _calculate_sensitivity(self) -> float:
        """Calculate L2 sensitivity for gradient updates"""
        # For gradient descent with gradient clipping, sensitivity equals max_grad_norm
        return self.max_grad_norm
        
    def _calculate_noise_scale(self) -> float:
        """Calculate noise scale based on privacy parameters"""
        # For Gaussian mechanism: σ = (sensitivity * noise_multiplier) / epsilon
        return (self.sensitivity * self.noise_multiplier) / self.epsilon
        
    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity"""
        clipped_gradients = {}
        
        # Calculate total gradient norm
        total_norm = 0.0
        for param_name, grad in gradients.items():
            if grad is not None:
                total_norm += torch.norm(grad) ** 2
        total_norm = torch.sqrt(total_norm)
        
        # Apply clipping
        if total_norm > self.max_grad_norm:
            clip_factor = self.max_grad_norm / total_norm
            for param_name, grad in gradients.items():
                if grad is not None:
                    clipped_gradients[param_name] = grad * clip_factor
                else:
                    clipped_gradients[param_name] = grad
        else:
            clipped_gradients = gradients
            
        return clipped_gradients
        
    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add calibrated Gaussian noise to gradients"""
        noisy_gradients = {}
        
        for param_name, grad in gradients.items():
            if grad is not None:
                # Generate Gaussian noise with appropriate scale
                noise = torch.normal(
                    mean=0.0,
                    std=self.noise_scale,
                    size=grad.shape,
                    device=grad.device
                )
                noisy_gradients[param_name] = grad + noise
            else:
                noisy_gradients[param_name] = grad
                
        return noisy_gradients
        
    def add_noise_to_model(self, model_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to model parameters"""
        noisy_params = {}
        
        for network_type, network_params in model_params.items():
            noisy_params[network_type] = {}
            
            for param_name, param_tensor in network_params.items():
                # Calculate parameter-specific noise scale
                param_noise_scale = self.noise_scale * math.sqrt(param_tensor.numel())
                
                # Generate and add noise
                noise = torch.normal(
                    mean=0.0,
                    std=param_noise_scale,
                    size=param_tensor.shape,
                    device=param_tensor.device
                )
                
                noisy_params[network_type][param_name] = param_tensor + noise
                
        # Update privacy accounting
        self._update_privacy_accounting()
        
        return noisy_params
        
    def add_noise_to_aggregation(self, aggregated_updates: Dict[str, torch.Tensor],
                                num_participants: int) -> Dict[str, torch.Tensor]:
        """Add noise to aggregated model updates"""
        # Scale noise by number of participants for better privacy/utility trade-off
        scaled_noise_scale = self.noise_scale / math.sqrt(num_participants)
        
        noisy_updates = {}
        for network_type, network_params in aggregated_updates.items():
            noisy_updates[network_type] = {}
            
            for param_name, param_tensor in network_params.items():
                noise = torch.normal(
                    mean=0.0,
                    std=scaled_noise_scale,
                    size=param_tensor.shape,
                    device=param_tensor.device
                )
                
                noisy_updates[network_type][param_name] = param_tensor + noise
                
        self._update_privacy_accounting()
        return noisy_updates
        
    def moment_accountant_privacy_spent(self, steps: int, sampling_rate: float) -> float:
        """Calculate privacy spent using moment accountant method"""
        if steps == 0:
            return 0.0
            
        # Simplified moment accountant calculation
        # In practice, you might want to use more sophisticated methods
        c = sampling_rate * math.sqrt(steps * math.log(1 / self.delta))
        return c * self.noise_multiplier / self.epsilon
        
    def _update_privacy_accounting(self):
        """Update privacy budget accounting"""
        self.composition_steps += 1
        
        # Simple composition (can be improved with advanced composition theorems)
        step_privacy_cost = self.epsilon / 100  # Conservative estimate
        self.privacy_spent += step_privacy_cost
        
        # Store privacy history
        self.privacy_history.append({
            'step': self.composition_steps,
            'privacy_spent': self.privacy_spent,
            'remaining_budget': max(0, self.epsilon - self.privacy_spent)
        })
        
        if self.privacy_spent >= self.epsilon:
            self.logger.warning(f"Privacy budget exhausted: {self.privacy_spent:.4f} >= {self.epsilon}")
            
    def get_privacy_spent(self) -> float:
        """Get total privacy budget spent"""
        return self.privacy_spent
        
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.epsilon - self.privacy_spent)
        
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        return self.privacy_spent >= self.epsilon
        
    def adaptive_noise_scale(self, current_round: int, total_rounds: int) -> float:
        """Adapt noise scale based on remaining rounds and budget"""
        remaining_rounds = total_rounds - current_round
        remaining_budget = self.get_remaining_budget()
        
        if remaining_rounds <= 0 or remaining_budget <= 0:
            return float('inf')  # No more privacy budget
            
        # Distribute remaining budget across remaining rounds
        budget_per_round = remaining_budget / remaining_rounds
        adaptive_epsilon = min(budget_per_round, self.epsilon / 10)  # Conservative
        
        return (self.sensitivity * self.noise_multiplier) / adaptive_epsilon
        
    def privacy_amplification_subsampling(self, sampling_rate: float) -> float:
        """Calculate privacy amplification due to subsampling"""
        # Privacy amplification by subsampling
        if sampling_rate >= 1.0:
            return self.epsilon
        else:
            # Simplified amplification (exact calculation is more complex)
            return self.epsilon * sampling_rate
            
    def calibrate_noise_for_target_epsilon(self, target_epsilon: float, 
                                         num_steps: int) -> float:
        """Calibrate noise scale for target epsilon over multiple steps"""
        # Account for composition over multiple steps
        per_step_epsilon = target_epsilon / math.sqrt(num_steps)
        return (self.sensitivity * self.noise_multiplier) / per_step_epsilon
        
    def dp_sgd_privacy_analysis(self, num_epochs: int, batch_size: int, 
                               dataset_size: int) -> Dict[str, float]:
        """Analyze privacy for DP-SGD-like training"""
        sampling_rate = batch_size / dataset_size
        num_steps = num_epochs * (dataset_size // batch_size)
        
        # Simplified analysis (in practice, use RDP accountant)
        amplified_epsilon = self.privacy_amplification_subsampling(sampling_rate)
        total_privacy_cost = amplified_epsilon * math.sqrt(num_steps)
        
        return {
            'sampling_rate': sampling_rate,
            'num_steps': num_steps,
            'amplified_epsilon': amplified_epsilon,
            'total_privacy_cost': total_privacy_cost,
            'meets_budget': total_privacy_cost <= self.epsilon
        }
        
    def privacy_report(self) -> Dict:
        """Generate comprehensive privacy report"""
        return {
            'privacy_parameters': {
                'epsilon': self.epsilon,
                'delta': self.delta,
                'noise_multiplier': self.noise_multiplier,
                'max_grad_norm': self.max_grad_norm
            },
            'privacy_accounting': {
                'privacy_spent': self.privacy_spent,
                'remaining_budget': self.get_remaining_budget(),
                'composition_steps': self.composition_steps,
                'budget_exhausted': self.is_budget_exhausted()
            },
            'noise_calibration': {
                'sensitivity': self.sensitivity,
                'noise_scale': self.noise_scale
            },
            'privacy_history': self.privacy_history[-10:] if self.privacy_history else []
        }
        
    def validate_privacy_parameters(self) -> bool:
        """Validate privacy parameter configuration"""
        valid = True
        
        if self.epsilon <= 0:
            self.logger.error("Epsilon must be positive")
            valid = False
            
        if self.delta <= 0 or self.delta >= 1:
            self.logger.error("Delta must be in (0, 1)")
            valid = False
            
        if self.noise_multiplier <= 0:
            self.logger.error("Noise multiplier must be positive")
            valid = False
            
        if self.max_grad_norm <= 0:
            self.logger.error("Max gradient norm must be positive")
            valid = False
            
        # Check if parameters provide meaningful privacy
        if self.epsilon > 10:
            self.logger.warning("Large epsilon may not provide meaningful privacy")
            
        if self.noise_multiplier < 0.1:
            self.logger.warning("Small noise multiplier may not provide sufficient privacy")
            
        return valid
        
    def reset(self):
        """Reset privacy accounting"""
        self.privacy_spent = 0.0
        self.composition_steps = 0
        self.privacy_history = []
        
        # Recalculate noise parameters
        self.sensitivity = self._calculate_sensitivity()
        self.noise_scale = self._calculate_noise_scale()
        
        self.logger.info("Privacy manager reset completed")
        
    def update_parameters(self, epsilon: Optional[float] = None,
                         delta: Optional[float] = None,
                         noise_multiplier: Optional[float] = None):
        """Update privacy parameters"""
        if epsilon is not None:
            self.epsilon = epsilon
        if delta is not None:
            self.delta = delta
        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
            
        # Recalculate dependent parameters
        self.sensitivity = self._calculate_sensitivity()
        self.noise_scale = self._calculate_noise_scale()
        
        # Validate new parameters
        if not self.validate_privacy_parameters():
            self.logger.error("Invalid privacy parameters after update")
            
        self.logger.info(f"Updated privacy parameters: ε={self.epsilon}, δ={self.delta}, "
                        f"noise_mult={self.noise_multiplier}")
