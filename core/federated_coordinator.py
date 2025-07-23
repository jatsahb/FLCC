"""
Federated Coordinator for NDN-FDRL System
Manages federated learning aggregation across network domains with privacy preservation
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import copy
import logging
from collections import defaultdict
import json
from datetime import datetime
import time

from utils.privacy import DifferentialPrivacyManager

class FederatedCoordinator:
    """Federated learning coordinator for multi-domain NDN network"""
    
    def __init__(self, domains: int, config: Dict):
        self.domains = domains
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Privacy manager
        self.privacy_manager = DifferentialPrivacyManager(
            epsilon=config.get('privacy_budget', 1.0),
            delta=config.get('privacy_delta', 1e-5),
            noise_multiplier=config.get('noise_multiplier', 1.1)
        )
        
        # Federated learning parameters
        self.aggregation_frequency = config.get('aggregation_frequency', 10)
        self.min_participants = config.get('min_participants', 2)
        self.convergence_threshold = config.get('convergence_threshold', 0.01)
        
        # Tracking variables
        self.global_model = None
        self.round_number = 0
        self.aggregation_history = []
        self.domain_participation = defaultdict(int)
        self.model_convergence = []
        
        # Performance metrics
        self.federation_metrics = {
            'communication_cost': 0,
            'aggregation_time': [],
            'model_accuracy': [],
            'privacy_budget_used': 0.0,
            'participant_diversity': []
        }
        
        self.logger = logging.getLogger("FederatedCoordinator")
        self.logger.info(f"Initialized federated coordinator for {domains} domains")
        
    def should_aggregate(self, episode: int) -> bool:
        """Determine if federated aggregation should occur"""
        return episode % self.aggregation_frequency == 0 and episode > 0
        
    def collect_models(self, agents: Dict[int, Any], episode: int) -> Dict[int, Dict]:
        """Collect models from agents organized by domain"""
        domain_models = defaultdict(list)
        
        for agent_id, agent in agents.items():
            domain = agent_id // (len(agents) // self.domains)
            
            # Get model parameters
            model_params = agent.get_model_parameters()
            
            # Calculate model quality metrics
            performance = agent.get_performance_summary()
            quality_score = self.calculate_model_quality(performance)
            
            domain_models[domain].append({
                'agent_id': agent_id,
                'parameters': model_params,
                'quality_score': quality_score,
                'performance': performance
            })
            
        self.logger.info(f"Collected models from {len(agents)} agents across {len(domain_models)} domains")
        return dict(domain_models)
        
    def calculate_model_quality(self, performance: Dict) -> float:
        """Calculate model quality score for weighted aggregation"""
        if not performance:
            return 0.5  # Default medium quality
            
        # Combine multiple performance metrics
        reward_score = min(1.0, max(0.0, (performance.get('avg_reward', 0) + 1) / 2))
        loss_score = 1.0 / (1.0 + performance.get('avg_critic_loss', 1.0))
        buffer_score = min(1.0, performance.get('replay_buffer_size', 0) / 10000)
        
        # Weighted combination
        quality = 0.5 * reward_score + 0.3 * loss_score + 0.2 * buffer_score
        return np.clip(quality, 0.1, 1.0)  # Ensure minimum quality
        
    def federated_averaging(self, domain_models: Dict[int, List[Dict]]) -> Dict:
        """Perform federated averaging with differential privacy"""
        start_time = datetime.now()
        
        # Filter domains with sufficient participants
        participating_domains = {
            domain: models for domain, models in domain_models.items()
            if len(models) >= self.min_participants
        }
        
        if not participating_domains:
            self.logger.warning("No domains have sufficient participants for aggregation")
            return None
            
        # Aggregate models within each domain first (intra-domain aggregation)
        domain_aggregates = {}
        for domain, models in participating_domains.items():
            domain_aggregate = self.aggregate_domain_models(models)
            domain_aggregates[domain] = domain_aggregate
            self.domain_participation[domain] += 1
            
        # Global aggregation across domains (inter-domain aggregation)
        global_model = self.aggregate_global_models(domain_aggregates)
        
        # Apply differential privacy
        if self.config.get('enable_privacy', True):
            global_model = self.privacy_manager.add_noise_to_model(global_model)
            self.federation_metrics['privacy_budget_used'] = self.privacy_manager.get_privacy_spent()
            
        # Update tracking
        self.round_number += 1
        aggregation_time = (datetime.now() - start_time).total_seconds()
        self.federation_metrics['aggregation_time'].append(aggregation_time)
        self.federation_metrics['communication_cost'] += len(participating_domains) * 2  # Upload + download
        
        # Calculate convergence metrics
        convergence_metric = self.calculate_convergence(global_model)
        self.model_convergence.append(convergence_metric)
        
        # Store aggregation history
        aggregation_record = {
            'round': self.round_number,
            'participating_domains': list(participating_domains.keys()),
            'participants_count': sum(len(models) for models in participating_domains.values()),
            'aggregation_time': aggregation_time,
            'convergence_metric': convergence_metric,
            'privacy_budget_used': self.federation_metrics['privacy_budget_used']
        }
        self.aggregation_history.append(aggregation_record)
        
        self.logger.info(f"Federated aggregation round {self.round_number} completed: "
                        f"{len(participating_domains)} domains, "
                        f"convergence: {convergence_metric:.4f}")
        
        self.global_model = global_model
        return global_model
        
    def aggregate_domain_models(self, models: List[Dict]) -> Dict:
        """Aggregate models within a single domain using quality-weighted averaging"""
        if not models:
            return None
            
        # Calculate weights based on quality scores
        quality_scores = [model['quality_score'] for model in models]
        total_quality = sum(quality_scores)
        
        if total_quality == 0:
            weights = [1.0 / len(models)] * len(models)
        else:
            weights = [score / total_quality for score in quality_scores]
            
        # Initialize aggregated parameters
        aggregated_params = {}
        first_model = models[0]['parameters']
        
        for network_type in first_model.keys():  # 'actor', 'critic'
            aggregated_params[network_type] = {}
            
            for param_name in first_model[network_type].keys():
                # Weighted average of parameters
                weighted_params = []
                for i, model in enumerate(models):
                    param = model['parameters'][network_type][param_name]
                    weighted_params.append(param * weights[i])
                    
                aggregated_params[network_type][param_name] = sum(weighted_params)
                
        return aggregated_params
        
    def aggregate_global_models(self, domain_aggregates: Dict[int, Dict]) -> Dict:
        """Aggregate domain models into global model"""
        if not domain_aggregates:
            return None
            
        # Equal weight for each domain (can be enhanced with domain-specific weights)
        num_domains = len(domain_aggregates)
        weight_per_domain = 1.0 / num_domains
        
        # Initialize global parameters
        global_params = {}
        first_domain = list(domain_aggregates.values())[0]
        
        for network_type in first_domain.keys():  # 'actor', 'critic'
            global_params[network_type] = {}
            
            for param_name in first_domain[network_type].keys():
                # Average across domains
                domain_params = [
                    domain_aggregate[network_type][param_name] * weight_per_domain
                    for domain_aggregate in domain_aggregates.values()
                ]
                global_params[network_type][param_name] = sum(domain_params)
                
        return global_params
        
    def distribute_global_model(self, agents: Dict[int, Any], global_model: Dict):
        """Distribute global model to all participating agents"""
        if global_model is None:
            self.logger.warning("No global model to distribute")
            return
            
        distribution_count = 0
        for agent_id, agent in agents.items():
            try:
                agent.set_model_parameters(global_model)
                distribution_count += 1
            except Exception as e:
                self.logger.error(f"Failed to update agent {agent_id}: {e}")
                
        self.logger.info(f"Distributed global model to {distribution_count} agents")
        
        # Update communication cost
        self.federation_metrics['communication_cost'] += distribution_count
        
    def calculate_convergence(self, global_model: Dict) -> float:
        """Calculate model convergence metric"""
        if self.global_model is None or global_model is None:
            return 1.0  # First model or no model
            
        # Calculate parameter differences between current and previous global model
        total_diff = 0.0
        total_params = 0
        
        for network_type in global_model.keys():
            for param_name in global_model[network_type].keys():
                current_param = global_model[network_type][param_name]
                previous_param = self.global_model[network_type][param_name]
                
                # Calculate L2 norm difference
                diff = torch.norm(current_param - previous_param).item()
                total_diff += diff
                total_params += current_param.numel()
                
        # Normalize by number of parameters
        convergence_metric = total_diff / total_params if total_params > 0 else 0.0
        return convergence_metric
        
    def is_converged(self) -> bool:
        """Check if the federated model has converged"""
        if len(self.model_convergence) < 3:
            return False
            
        # Check if recent convergence metrics are below threshold
        recent_convergence = self.model_convergence[-3:]
        return all(metric < self.convergence_threshold for metric in recent_convergence)
        
    def get_federation_status(self) -> Dict:
        """Get current federation status and metrics"""
        status = {
            'round_number': self.round_number,
            'total_aggregations': len(self.aggregation_history),
            'participating_domains': len(self.domain_participation),
            'is_converged': self.is_converged(),
            'privacy_budget_remaining': self.privacy_manager.get_remaining_budget(),
            'average_participants': np.mean([
                record['participants_count'] for record in self.aggregation_history
            ]) if self.aggregation_history else 0,
            'latest_convergence': self.model_convergence[-1] if self.model_convergence else None
        }
        
        return status
        
    def get_detailed_metrics(self) -> Dict:
        """Get detailed federation metrics for analysis"""
        metrics = {
            'federation_metrics': self.federation_metrics.copy(),
            'aggregation_history': self.aggregation_history.copy(),
            'domain_participation': dict(self.domain_participation),
            'model_convergence': self.model_convergence.copy(),
            'current_status': self.get_federation_status()
        }
        
        # Calculate additional derived metrics
        if self.aggregation_history:
            metrics['communication_efficiency'] = (
                np.mean([r['participants_count'] for r in self.aggregation_history]) /
                max(1, np.mean([r['aggregation_time'] for r in self.aggregation_history]))
            )
            
            metrics['convergence_rate'] = (
                self.model_convergence[0] - self.model_convergence[-1]
            ) / len(self.model_convergence) if len(self.model_convergence) > 1 else 0
            
        return metrics
        
    def reset(self):
        """Reset coordinator state for new simulation"""
        self.global_model = None
        self.round_number = 0
        self.aggregation_history = []
        self.domain_participation = defaultdict(int)
        self.model_convergence = []
        
        # Reset metrics
        self.federation_metrics = {
            'communication_cost': 0,
            'aggregation_time': [],
            'model_accuracy': [],
            'privacy_budget_used': 0.0,
            'participant_diversity': []
        }
        
        # Reset privacy manager
        self.privacy_manager.reset()
        
        self.logger.info("Federated coordinator reset completed")
