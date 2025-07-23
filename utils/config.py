"""
Configuration Management for NDN-FDRL System
Centralized configuration management with validation and environment support
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

@dataclass
class NetworkConfig:
    """Network simulation configuration"""
    domains: int = 3
    nodes_per_domain: int = 5
    default_bandwidth: float = 10.0  # Mbps
    default_latency: float = 50.0    # ms
    max_queue_size: int = 1000
    cache_size: int = 1000
    pit_max_entries: int = 10000
    
@dataclass
class DDPGConfig:
    """DDPG agent configuration"""
    state_dim: int = 8
    action_dim: int = 4
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    batch_size: int = 64
    memory_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    noise_theta: float = 0.15
    noise_sigma: float = 0.2
    
@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    aggregation_frequency: int = 10
    min_participants: int = 2
    convergence_threshold: float = 0.01
    max_rounds: int = 100
    participation_rate: float = 1.0
    
@dataclass
class PrivacyConfig:
    """Differential privacy configuration"""
    enable_privacy: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
@dataclass
class ScenarioConfig:
    """Scenario generation configuration"""
    default_scenario_id: int = 1
    scenario_duration: int = 300  # seconds
    severity_levels: Dict[str, tuple] = None
    enable_dynamic_scenarios: bool = True
    
    def __post_init__(self):
        if self.severity_levels is None:
            self.severity_levels = {
                'low': (0.3, 0.5),
                'medium': (0.5, 0.7),
                'high': (0.7, 0.9),
                'critical': (0.9, 1.0)
            }
            
@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    log_interval: int = 10  # seconds
    enable_real_time_monitoring: bool = True
    alert_thresholds: Dict[str, Dict[str, float]] = None
    export_format: str = 'json'
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'congestion_level': {'high': 0.8, 'critical': 0.9},
                'packet_loss_rate': {'high': 0.1, 'critical': 0.2},
                'latency': {'high': 200, 'critical': 500},
                'cache_hit_rate': {'low': 0.3, 'critical': 0.2}
            }

class SimulationConfig:
    """Main simulation configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger("SimulationConfig")
        
        # Initialize default configurations
        self.network = NetworkConfig()
        self.ddpg = DDPGConfig()
        self.federated = FederatedConfig()
        self.privacy = PrivacyConfig()
        self.scenario = ScenarioConfig()
        self.monitoring = MonitoringConfig()
        
        # Environment-based overrides
        self._load_environment_variables()
        
        # Load from config file if provided
        if config_file:
            self.load_from_file(config_file)
            
        # Validate configuration
        self.validate_config()
        
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        # Network configuration
        self.network.domains = int(os.getenv('NDN_DOMAINS', self.network.domains))
        self.network.nodes_per_domain = int(os.getenv('NDN_NODES_PER_DOMAIN', self.network.nodes_per_domain))
        self.network.default_bandwidth = float(os.getenv('NDN_DEFAULT_BANDWIDTH', self.network.default_bandwidth))
        
        # DDPG configuration
        self.ddpg.learning_rate = float(os.getenv('DDPG_LEARNING_RATE', self.ddpg.learning_rate))
        self.ddpg.batch_size = int(os.getenv('DDPG_BATCH_SIZE', self.ddpg.batch_size))
        self.ddpg.hidden_dim = int(os.getenv('DDPG_HIDDEN_DIM', self.ddpg.hidden_dim))
        
        # Federated learning configuration
        self.federated.aggregation_frequency = int(os.getenv('FL_AGGREGATION_FREQ', self.federated.aggregation_frequency))
        self.federated.min_participants = int(os.getenv('FL_MIN_PARTICIPANTS', self.federated.min_participants))
        
        # Privacy configuration
        self.privacy.enable_privacy = os.getenv('DP_ENABLE', 'true').lower() == 'true'
        self.privacy.epsilon = float(os.getenv('DP_EPSILON', self.privacy.epsilon))
        self.privacy.delta = float(os.getenv('DP_DELTA', self.privacy.delta))
        
        # Monitoring configuration
        self.monitoring.log_interval = int(os.getenv('LOG_INTERVAL', self.monitoring.log_interval))
        
    def load_from_file(self, config_file: str):
        """Load configuration from file (JSON or YAML)"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.logger.warning(f"Configuration file {config_file} not found, using defaults")
            return
            
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
                    
            self._update_from_dict(config_data)
            self.logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
            
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        if 'network' in config_data:
            self._update_dataclass(self.network, config_data['network'])
            
        if 'ddpg' in config_data:
            self._update_dataclass(self.ddpg, config_data['ddpg'])
            
        if 'federated' in config_data:
            self._update_dataclass(self.federated, config_data['federated'])
            
        if 'privacy' in config_data:
            self._update_dataclass(self.privacy, config_data['privacy'])
            
        if 'scenario' in config_data:
            self._update_dataclass(self.scenario, config_data['scenario'])
            
        if 'monitoring' in config_data:
            self._update_dataclass(self.monitoring, config_data['monitoring'])
            
    def _update_dataclass(self, dataclass_instance: Any, update_dict: Dict[str, Any]):
        """Update dataclass instance with dictionary values"""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
                
    def validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Network validation
        if self.network.domains <= 0:
            errors.append("Number of domains must be positive")
        if self.network.nodes_per_domain <= 0:
            errors.append("Nodes per domain must be positive")
        if self.network.default_bandwidth <= 0:
            errors.append("Default bandwidth must be positive")
            
        # DDPG validation
        if self.ddpg.state_dim <= 0:
            errors.append("DDPG state dimension must be positive")
        if self.ddpg.action_dim <= 0:
            errors.append("DDPG action dimension must be positive")
        if self.ddpg.learning_rate <= 0 or self.ddpg.learning_rate > 1:
            errors.append("DDPG learning rate must be in (0, 1]")
        if self.ddpg.gamma <= 0 or self.ddpg.gamma > 1:
            errors.append("DDPG gamma must be in (0, 1]")
            
        # Federated learning validation
        if self.federated.aggregation_frequency <= 0:
            errors.append("Aggregation frequency must be positive")
        if self.federated.min_participants <= 0:
            errors.append("Minimum participants must be positive")
        if self.federated.min_participants > self.network.domains * self.network.nodes_per_domain:
            errors.append("Minimum participants cannot exceed total nodes")
            
        # Privacy validation
        if self.privacy.enable_privacy:
            if self.privacy.epsilon <= 0:
                errors.append("Privacy epsilon must be positive")
            if self.privacy.delta <= 0 or self.privacy.delta >= 1:
                errors.append("Privacy delta must be in (0, 1)")
            if self.privacy.noise_multiplier <= 0:
                errors.append("Noise multiplier must be positive")
                
        # Scenario validation
        if self.scenario.scenario_duration <= 0:
            errors.append("Scenario duration must be positive")
            
        # Monitoring validation
        if self.monitoring.log_interval <= 0:
            errors.append("Log interval must be positive")
            
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            self.logger.info("Configuration validation passed")
            
    def get_network_config(self) -> Dict[str, Any]:
        """Get network configuration as dictionary"""
        return asdict(self.network)
        
    def get_ddpg_config(self) -> Dict[str, Any]:
        """Get DDPG configuration as dictionary"""
        return asdict(self.ddpg)
        
    def get_federated_config(self) -> Dict[str, Any]:
        """Get federated learning configuration as dictionary"""
        return asdict(self.federated)
        
    def get_privacy_config(self) -> Dict[str, Any]:
        """Get privacy configuration as dictionary"""
        return asdict(self.privacy)
        
    def get_scenario_config(self) -> Dict[str, Any]:
        """Get scenario configuration as dictionary"""
        return asdict(self.scenario)
        
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration as dictionary"""
        return asdict(self.monitoring)
        
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'network': self.get_network_config(),
            'ddpg': self.get_ddpg_config(),
            'federated': self.get_federated_config(),
            'privacy': self.get_privacy_config(),
            'scenario': self.get_scenario_config(),
            'monitoring': self.get_monitoring_config()
        }
        
    def save_to_file(self, config_file: str, format: str = 'json'):
        """Save current configuration to file"""
        config_path = Path(config_file)
        config_data = self.get_all_config()
        
        try:
            with open(config_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, default=str)
                    
            self.logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_file}: {e}")
            
    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update specific configuration section"""
        section_map = {
            'network': self.network,
            'ddpg': self.ddpg,
            'federated': self.federated,
            'privacy': self.privacy,
            'scenario': self.scenario,
            'monitoring': self.monitoring
        }
        
        if section not in section_map:
            raise ValueError(f"Unknown configuration section: {section}")
            
        self._update_dataclass(section_map[section], updates)
        self.validate_config()
        
        self.logger.info(f"Updated {section} configuration")
        
    def create_agent_config(self, agent_id: int) -> Dict[str, Any]:
        """Create agent-specific configuration"""
        base_config = self.get_ddpg_config()
        
        # Add agent-specific parameters
        base_config['agent_id'] = agent_id
        base_config['device'] = 'cuda' if os.getenv('USE_GPU', 'false').lower() == 'true' else 'cpu'
        
        return base_config
        
    def create_federated_coordinator_config(self) -> Dict[str, Any]:
        """Create federated coordinator configuration"""
        config = self.get_federated_config()
        config.update(self.get_privacy_config())
        config['domains'] = self.network.domains
        
        return config
        
    def create_simulator_config(self) -> Dict[str, Any]:
        """Create simulator configuration"""
        config = self.get_network_config()
        config.update(self.get_scenario_config())
        
        return config
        
    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration for the simulation"""
        return {
            'total_nodes': self.network.domains * self.network.nodes_per_domain,
            'simulation_duration': self.scenario.scenario_duration,
            'log_interval': self.monitoring.log_interval,
            'enable_privacy': self.privacy.enable_privacy,
            'enable_federated_learning': True,
            'real_time_monitoring': self.monitoring.enable_real_time_monitoring
        }
        
    def __str__(self) -> str:
        """String representation of configuration"""
        config_summary = [
            f"NDN-FDRL Configuration:",
            f"  Network: {self.network.domains} domains, {self.network.nodes_per_domain} nodes/domain",
            f"  DDPG: lr={self.ddpg.learning_rate}, hidden={self.ddpg.hidden_dim}",
            f"  Federated: freq={self.federated.aggregation_frequency}, min_participants={self.federated.min_participants}",
            f"  Privacy: enabled={self.privacy.enable_privacy}, ε={self.privacy.epsilon}",
            f"  Monitoring: interval={self.monitoring.log_interval}s"
        ]
        return "\n".join(config_summary)
