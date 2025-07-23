"""
Scenario Generator for NDN Networks
Creates realistic congestion and failure scenarios for testing federated learning effectiveness
"""

import random
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

class ScenarioGenerator:
    """Advanced congestion scenario generator for NDN networks"""
    
    def __init__(self, max_scenarios: int = 20):
        self.max_scenarios = max_scenarios
        self.scenario_templates = self.load_scenario_templates()
        self.scenario_history = []
        self.active_scenarios = {}
        
        # Scenario parameters
        self.severity_distribution = {
            'low': (0.3, 0.5),
            'medium': (0.5, 0.7),
            'high': (0.7, 0.9),
            'critical': (0.9, 1.0)
        }
        
        # Temporal patterns
        self.time_patterns = {
            'flash_crowd': {'duration': (30, 120), 'ramp_up': 10},
            'sustained': {'duration': (300, 600), 'ramp_up': 60},
            'intermittent': {'duration': (60, 180), 'ramp_up': 30},
            'gradual': {'duration': (180, 300), 'ramp_up': 120}
        }
        
        self.logger = logging.getLogger("ScenarioGenerator")
        
    def load_scenario_templates(self) -> Dict:
        """Load scenario templates with realistic NDN congestion patterns"""
        return {
            1: {
                'name': 'Flash Crowd',
                'description': 'Sudden spike in requests for popular content',
                'type': 'demand_surge',
                'affected_components': ['consumers', 'content_servers'],
                'primary_metrics': ['request_rate', 'cache_miss_ratio'],
                'trigger_probability': 0.15
            },
            2: {
                'name': 'Link Failure',
                'description': 'Critical network link becomes unavailable',
                'type': 'infrastructure_failure',
                'affected_components': ['routers', 'paths'],
                'primary_metrics': ['latency', 'packet_loss', 'path_availability'],
                'trigger_probability': 0.08
            },
            3: {
                'name': 'Cache Poisoning Attack',
                'description': 'Malicious content injection into network caches',
                'type': 'security_attack',
                'affected_components': ['content_stores', 'routers'],
                'primary_metrics': ['cache_pollution', 'verification_overhead'],
                'trigger_probability': 0.05
            },
            4: {
                'name': 'DDoS Interest Flooding',
                'description': 'Massive volume of malicious interest packets',
                'type': 'security_attack',
                'affected_components': ['routers', 'pit_tables'],
                'primary_metrics': ['pit_overflow', 'processing_overhead'],
                'trigger_probability': 0.06
            },
            5: {
                'name': 'Producer Overload',
                'description': 'Content producers become overwhelmed with requests',
                'type': 'resource_exhaustion',
                'affected_components': ['producers', 'origin_servers'],
                'primary_metrics': ['response_time', 'rejection_rate'],
                'trigger_probability': 0.10
            }
        }
        
    def generate_scenario(self, scenario_id: Optional[int] = None, 
                         severity_level: Optional[str] = None,
                         duration_pattern: Optional[str] = None) -> Dict:
        """Generate a comprehensive congestion scenario"""
        
        # Select scenario
        if scenario_id is None:
            # Probabilistic selection based on trigger probabilities
            probabilities = [template['trigger_probability'] 
                           for template in self.scenario_templates.values()]
            scenario_id = random.choices(
                list(self.scenario_templates.keys()), 
                weights=probabilities
            )[0]
        elif scenario_id not in self.scenario_templates:
            scenario_id = random.choice(list(self.scenario_templates.keys()))
            
        template = self.scenario_templates[scenario_id]
        
        # Determine severity
        if severity_level is None:
            severity_level = random.choices(
                list(self.severity_distribution.keys()),
                weights=[0.3, 0.4, 0.25, 0.05]  # Bias towards medium severity
            )[0]
            
        severity_range = self.severity_distribution[severity_level]
        severity = random.uniform(*severity_range)
        
        # Determine temporal pattern
        if duration_pattern is None:
            duration_pattern = random.choice(list(self.time_patterns.keys()))
            
        time_config = self.time_patterns[duration_pattern]
        duration = random.randint(*time_config['duration'])
        ramp_up_time = time_config['ramp_up']
        
        # Generate scenario-specific parameters
        scenario_params = self.generate_scenario_parameters(
            scenario_id, severity, template
        )
        
        # Create comprehensive scenario description
        scenario = {
            'id': scenario_id,
            'name': template['name'],
            'description': template['description'],
            'type': template['type'],
            'severity_level': severity_level,
            'severity': severity,
            'duration': duration,
            'duration_pattern': duration_pattern,
            'ramp_up_time': ramp_up_time,
            'affected_components': template['affected_components'],
            'primary_metrics': template['primary_metrics'],
            'parameters': scenario_params,
            'timestamp': datetime.now(),
            'status': 'generated'
        }
        
        self.scenario_history.append(scenario)
        self.logger.info(f"Generated scenario: {template['name']} "
                        f"(ID: {scenario_id}, Severity: {severity:.2f})")
        
        return scenario
        
    def generate_scenario_parameters(self, scenario_id: int, severity: float, 
                                   template: Dict) -> Dict:
        """Generate scenario-specific parameters based on type and severity"""
        
        base_params = {
            'severity_multiplier': severity,
            'affected_node_ratio': min(1.0, 0.1 + severity * 0.6),
            'impact_radius': int(1 + severity * 5)
        }
        
        if scenario_id == 1:  # Flash Crowd
            return {
                **base_params,
                'request_multiplier': 2 + severity * 8,
                'content_concentration': 0.8 + severity * 0.15,
                'geographic_concentration': random.uniform(0.3, 0.8),
                'peak_duration': int(30 + severity * 90),
                'cache_overflow_threshold': 0.7 + severity * 0.25
            }
            
        elif scenario_id == 2:  # Link Failure
            return {
                **base_params,
                'failure_cascade_probability': severity * 0.4,
                'recovery_time': int(60 + severity * 240),
                'rerouting_overhead': 1.2 + severity * 2.0,
                'affected_paths': int(1 + severity * 4),
                'backup_path_quality': max(0.1, 1.0 - severity * 0.6)
            }
            
        elif scenario_id == 3:  # Cache Poisoning
            return {
                **base_params,
                'poisoned_content_ratio': severity * 0.3,
                'verification_overhead': 1.5 + severity * 3.0,
                'detection_delay': int(20 + severity * 180),
                'cleanup_duration': int(60 + severity * 300),
                'false_positive_rate': severity * 0.1
            }
            
        elif scenario_id == 4:  # DDoS Interest Flooding
            return {
                **base_params,
                'attack_interest_rate': int(1000 + severity * 9000),
                'pit_overflow_threshold': 0.8 + severity * 0.15,
                'legitimate_traffic_impact': severity * 0.7,
                'attack_duration': int(60 + severity * 300),
                'source_diversity': int(5 + severity * 20)
            }
            
        elif scenario_id == 5:  # Producer Overload
            return {
                **base_params,
                'overload_threshold': 0.6 + severity * 0.3,
                'response_delay_multiplier': 1.5 + severity * 4.0,
                'rejection_rate': severity * 0.4,
                'recovery_rate': max(0.1, 1.0 - severity * 0.5),
                'load_balancing_effectiveness': max(0.2, 1.0 - severity * 0.4)
            }
            
        return base_params
        
    def apply_scenario_to_network(self, scenario: Dict, network_state: Dict, 
                                timestep: int) -> Dict:
        """Apply scenario effects to network state"""
        if not scenario or scenario['status'] != 'active':
            return network_state
            
        scenario_type = scenario['type']
        params = scenario['parameters']
        severity = params['severity_multiplier']
        
        # Calculate time-based intensity (ramp up, sustain, ramp down)
        scenario_age = timestep - scenario.get('start_timestep', 0)
        duration = scenario['duration']
        ramp_up = scenario['ramp_up_time']
        
        if scenario_age < ramp_up:
            intensity = (scenario_age / ramp_up) * severity
        elif scenario_age < duration - ramp_up:
            intensity = severity
        else:
            remaining_time = duration - scenario_age
            intensity = (remaining_time / ramp_up) * severity if remaining_time > 0 else 0
            
        # Apply type-specific effects
        modified_state = network_state.copy()
        
        if scenario_type == 'demand_surge':
            modified_state = self.apply_demand_surge_effects(
                modified_state, params, intensity
            )
        elif scenario_type == 'infrastructure_failure':
            modified_state = self.apply_infrastructure_failure_effects(
                modified_state, params, intensity
            )
        elif scenario_type == 'security_attack':
            modified_state = self.apply_security_attack_effects(
                modified_state, params, intensity
            )
        elif scenario_type == 'resource_exhaustion':
            modified_state = self.apply_resource_exhaustion_effects(
                modified_state, params, intensity
            )
            
        return modified_state
        
    def apply_demand_surge_effects(self, network_state: Dict, 
                                 params: Dict, intensity: float) -> Dict:
        """Apply demand surge scenario effects"""
        request_multiplier = 1.0 + (params['request_multiplier'] - 1.0) * intensity
        
        # Increase request rates at consumers
        for node_id, node_state in network_state.get('nodes', {}).items():
            if node_state.get('type') == 'consumer':
                node_state['request_rate'] = node_state.get('request_rate', 10) * request_multiplier
                
        # Increase cache miss rates due to content diversity
        cache_miss_increase = intensity * params['content_concentration']
        for node_id, node_state in network_state.get('nodes', {}).items():
            if node_state.get('type') in ['router', 'producer']:
                current_hit_rate = node_state.get('cache_hit_rate', 0.5)
                node_state['cache_hit_rate'] = max(0.1, current_hit_rate * (1 - cache_miss_increase))
                
        return network_state
        
    def apply_infrastructure_failure_effects(self, network_state: Dict,
                                           params: Dict, intensity: float) -> Dict:
        """Apply infrastructure failure scenario effects"""
        failure_impact = intensity * params['failure_cascade_probability']
        
        # Increase latency due to rerouting
        rerouting_overhead = 1.0 + (params['rerouting_overhead'] - 1.0) * intensity
        
        for node_id, node_state in network_state.get('nodes', {}).items():
            if node_state.get('type') == 'router':
                node_state['latency'] = node_state.get('latency', 50) * rerouting_overhead
                node_state['packet_loss_rate'] = min(0.3, 
                    node_state.get('packet_loss_rate', 0.01) + failure_impact * 0.1
                )
                
        return network_state
        
    def apply_security_attack_effects(self, network_state: Dict,
                                    params: Dict, intensity: float) -> Dict:
        """Apply security attack scenario effects"""
        attack_intensity = intensity * params.get('attack_interest_rate', 5000) / 10000
        
        # Increase processing overhead and congestion
        for node_id, node_state in network_state.get('nodes', {}).items():
            if node_state.get('type') in ['router', 'producer']:
                node_state['congestion_level'] = min(1.0,
                    node_state.get('congestion_level', 0.2) + attack_intensity * 0.5
                )
                node_state['processing_overhead'] = 1.0 + attack_intensity * 2.0
                
        return network_state
        
    def apply_resource_exhaustion_effects(self, network_state: Dict,
                                        params: Dict, intensity: float) -> Dict:
        """Apply resource exhaustion scenario effects"""
        overload_impact = intensity * params['overload_threshold']
        
        # Increase response delays and rejection rates at producers
        for node_id, node_state in network_state.get('nodes', {}).items():
            if node_state.get('type') == 'producer':
                delay_multiplier = 1.0 + (params['response_delay_multiplier'] - 1.0) * intensity
                node_state['response_time'] = node_state.get('response_time', 10) * delay_multiplier
                node_state['rejection_rate'] = min(0.5, 
                    node_state.get('rejection_rate', 0.02) + overload_impact * 0.3
                )
                
        return network_state
        
    def get_scenario_summary(self, scenario: Dict) -> Dict:
        """Get a summary of scenario characteristics"""
        return {
            'id': scenario['id'],
            'name': scenario['name'],
            'type': scenario['type'],
            'severity': scenario['severity'],
            'duration': scenario['duration'],
            'affected_components': scenario['affected_components'],
            'key_parameters': {
                k: v for k, v in scenario['parameters'].items()
                if k in ['severity_multiplier', 'affected_node_ratio', 'impact_radius']
            }
        }
        
    def get_all_scenarios(self) -> List[Dict]:
        """Get all available scenario templates"""
        scenarios = []
        for scenario_id, template in self.scenario_templates.items():
            scenarios.append({
                'id': scenario_id,
                'name': template['name'],
                'description': template['description'],
                'type': template['type'],
                'affected_components': template['affected_components'],
                'trigger_probability': template['trigger_probability']
            })
        return scenarios
        
    def reset(self):
        """Reset scenario generator state"""
        self.scenario_history = []
        self.active_scenarios = {}
        self.logger.info("Scenario generator reset completed")
