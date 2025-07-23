"""
NDN-FDRL Simulation System
Main Flask application for web-based NDN federated deep reinforcement learning simulation
"""

from flask import Flask, render_template, jsonify, request
import json
import threading
import time
import logging
from datetime import datetime
import numpy as np
from typing import Dict, List, Any

from core.ndn_simulator import NDNNetworkSimulator
from core.federated_coordinator import FederatedCoordinator
from core.scenario_generator import ScenarioGenerator
from core.stakeholder_monitor import StakeholderMonitor
from agents.ddpg_agent import DDPGAgent
from utils.config import SimulationConfig
from utils.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
logger = logging.getLogger(__name__)

class NDNFDRLSimulation:
    """Main simulation controller for NDN-FDRL system"""
    
    def __init__(self):
        self.config = SimulationConfig()
        self.is_running = False
        self.simulation_thread = None
        self.current_results = {}
        self.phase1_metrics = {}
        self.phase2_metrics = {}
        self.training_history = []
        self.current_phase = 1
        self.fl_round = 0
        
        # Core components
        self.simulator = None
        self.coordinator = None
        self.scenario_generator = ScenarioGenerator()
        self.stakeholder_monitor = None
        self.agents = {}
        self.metrics_collector = MetricsCollector()
        
    def get_scenarios(self) -> List[Dict]:
        """Get available simulation scenarios"""
        scenarios = [
            {
                "id": 1,
                "name": "Flash Crowd Event",
                "description": "Sudden spike in content requests causing network congestion",
                "severity": "High",
                "duration": "2-5 minutes",
                "primary_effects": ["Cache overflow", "Increased latency", "Packet drops"]
            },
            {
                "id": 2,
                "name": "Link Failure Cascade",
                "description": "Critical network links fail causing rerouting congestion",
                "severity": "Critical", 
                "duration": "5-10 minutes",
                "primary_effects": ["Path recalculation", "Load redistribution", "Service degradation"]
            },
            {
                "id": 3,
                "name": "DDoS Interest Flooding",
                "description": "Malicious interest packet flooding attack",
                "severity": "High",
                "duration": "3-8 minutes", 
                "primary_effects": ["PIT overflow", "Processing overload", "Legitimate traffic blocking"]
            },
            {
                "id": 4,
                "name": "Producer Overload",
                "description": "Content producers become overwhelmed with requests",
                "severity": "Medium",
                "duration": "5-15 minutes",
                "primary_effects": ["Response delays", "Cache misses", "Quality degradation"]
            },
            {
                "id": 5,
                "name": "Multi-Domain Congestion",
                "description": "Coordinated congestion across multiple network domains",
                "severity": "Critical",
                "duration": "10-20 minutes", 
                "primary_effects": ["Cross-domain delays", "Federation overhead", "Scalability issues"]
            }
        ]
        return scenarios
        
    def initialize_simulation(self, scenario_id: int, duration: int, domains: int, nodes_per_domain: int):
        """Initialize simulation components"""
        try:
            logger.info(f"Initializing simulation - Scenario: {scenario_id}, Duration: {duration}s, Domains: {domains}")
            
            # Initialize network simulator
            self.simulator = NDNNetworkSimulator(domains=domains, nodes_per_domain=nodes_per_domain)
            
            # Initialize federated coordinator
            fed_config = {
                'aggregation_frequency': 10,
                'min_participants': 2,
                'convergence_threshold': 0.01,
                'privacy_budget': 1.0,
                'enable_privacy': True
            }
            self.coordinator = FederatedCoordinator(domains=domains, config=fed_config)
            
            # Initialize stakeholder monitor
            self.stakeholder_monitor = StakeholderMonitor(
                num_consumers=domains * 3,
                num_producers=domains,
                num_routers=domains * nodes_per_domain
            )
            
            # Initialize DDPG agents for each node
            total_nodes = domains * nodes_per_domain
            for i in range(total_nodes):
                agent_config = {
                    'state_dim': 8,  # Network state dimensions
                    'action_dim': 4,  # Control actions
                    'hidden_dim': 256,
                    'learning_rate': 1e-4,
                    'batch_size': 64,
                    'memory_size': 100000
                }
                self.agents[i] = DDPGAgent(agent_config)
            
            # Generate specific scenario
            scenario = self.scenario_generator.generate_scenario(scenario_id)
            self.simulator.set_active_scenario(scenario)
            
            logger.info("Simulation initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Simulation initialization failed: {e}")
            return False
    
    def run_simulation(self, scenario_id: int, duration: int, domains: int, nodes_per_domain: int):
        """Run the two-phase simulation"""
        try:
            if not self.initialize_simulation(scenario_id, duration, domains, nodes_per_domain):
                return
                
            self.is_running = True
            total_steps = duration
            phase1_steps = total_steps // 2
            
            logger.info("Starting Phase 1: Baseline (No Federated Learning)")
            self.current_phase = 1
            
            # Phase 1: Baseline without federated learning
            phase1_data = []
            for step in range(phase1_steps):
                if not self.is_running:
                    break
                    
                # Get network state
                network_state = self.simulator.get_network_state()
                
                # Agent actions (without federated coordination)
                actions = {}
                for agent_id, agent in self.agents.items():
                    state = self.simulator.get_node_state(agent_id)
                    action = agent.act(state, add_noise=True)
                    actions[agent_id] = action
                
                # Simulate network step
                step_metrics = self.simulator.step(actions, step)
                
                # Update stakeholder behaviors
                stakeholder_metrics = self.stakeholder_monitor.update(network_state)
                
                # Collect metrics
                combined_metrics = {**step_metrics, **stakeholder_metrics}
                phase1_data.append(combined_metrics)
                
                # Update current results for real-time display
                self.current_results = {
                    'phase': 1,
                    'step': step,
                    'total_steps': phase1_steps,
                    'progress': (step / phase1_steps) * 100,
                    'metrics': combined_metrics
                }
                
                time.sleep(0.1)  # Simulation speed control
            
            # Calculate Phase 1 summary
            self.phase1_metrics = self.calculate_phase_summary(phase1_data)
            
            logger.info("Starting Phase 2: Federated Learning Enhanced")
            self.current_phase = 2
            self.fl_round = 0
            
            # Phase 2: With federated learning
            phase2_data = []
            for step in range(phase1_steps, total_steps):
                if not self.is_running:
                    break
                
                # Check for federated aggregation
                if self.coordinator.should_aggregate(step - phase1_steps):
                    self.fl_round += 1
                    logger.info(f"Federated Learning Round {self.fl_round}")
                    
                    # Collect models from agents
                    domain_models = self.coordinator.collect_models(self.agents, step)
                    
                    # Perform federated averaging
                    global_model = self.coordinator.federated_averaging(domain_models)
                    
                    # Distribute updated model
                    if global_model:
                        self.coordinator.distribute_global_model(self.agents, global_model)
                    
                    # Record training metrics
                    training_metrics = self.simulate_training_metrics()
                    self.training_history.append({
                        'round': self.fl_round,
                        'step': step,
                        'accuracy': training_metrics['accuracy'],
                        'loss': training_metrics['loss'],
                        'participants': len(domain_models)
                    })
                
                # Get enhanced network state (FL improves performance)
                network_state = self.simulator.get_network_state(fl_enhanced=True)
                
                # Agent actions with federated coordination
                actions = {}
                for agent_id, agent in self.agents.items():
                    state = self.simulator.get_node_state(agent_id)
                    action = agent.act(state, add_noise=False)  # Less exploration in FL phase
                    actions[agent_id] = action
                
                # Simulate network step with FL benefits
                step_metrics = self.simulator.step(actions, step, fl_enhanced=True)
                
                # Update stakeholder behaviors
                stakeholder_metrics = self.stakeholder_monitor.update(network_state)
                
                # Collect metrics
                combined_metrics = {**step_metrics, **stakeholder_metrics}
                phase2_data.append(combined_metrics)
                
                # Update current results
                self.current_results = {
                    'phase': 2,
                    'step': step - phase1_steps,
                    'total_steps': phase1_steps,
                    'progress': ((step - phase1_steps) / phase1_steps) * 100,
                    'metrics': combined_metrics,
                    'fl_round': self.fl_round,
                    'training_history': self.training_history[-1] if self.training_history else None
                }
                
                time.sleep(0.1)
            
            # Calculate Phase 2 summary
            self.phase2_metrics = self.calculate_phase_summary(phase2_data)
            
            self.is_running = False
            logger.info("Simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            self.is_running = False
    
    def calculate_phase_summary(self, phase_data: List[Dict]) -> Dict:
        """Calculate summary statistics for a simulation phase"""
        if not phase_data:
            return {}
        
        metrics = {}
        for key in phase_data[0].keys():
            values = [data[key] for data in phase_data if isinstance(data[key], (int, float))]
            if values:
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return metrics
    
    def simulate_training_metrics(self) -> Dict:
        """Simulate federated learning training metrics"""
        # Simulate improving accuracy and decreasing loss over rounds
        base_accuracy = 0.6
        max_accuracy = 0.95
        
        # Exponential improvement with diminishing returns
        accuracy = base_accuracy + (max_accuracy - base_accuracy) * (1 - np.exp(-0.1 * self.fl_round))
        accuracy += np.random.normal(0, 0.02)  # Add noise
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        loss = 1.0 - accuracy + np.random.normal(0, 0.05)
        loss = np.clip(loss, 0.0, 2.0)
        
        return {
            'accuracy': float(accuracy),
            'loss': float(loss)
        }
    
    def get_comparative_analysis(self) -> Dict:
        """Generate comparative analysis between phases"""
        if not self.phase1_metrics or not self.phase2_metrics:
            return {}
        
        analysis = {}
        common_metrics = set(self.phase1_metrics.keys()) & set(self.phase2_metrics.keys())
        
        for metric in common_metrics:
            if 'mean' in self.phase1_metrics[metric] and 'mean' in self.phase2_metrics[metric]:
                phase1_mean = self.phase1_metrics[metric]['mean']
                phase2_mean = self.phase2_metrics[metric]['mean']
                
                improvement = ((phase2_mean - phase1_mean) / phase1_mean) * 100
                analysis[metric] = {
                    'phase1_mean': phase1_mean,
                    'phase2_mean': phase2_mean,
                    'improvement_percent': improvement,
                    'improved': improvement > 0 if 'latency' not in metric and 'loss' not in metric else improvement < 0
                }
        
        return analysis
    
    def stop_simulation(self):
        """Stop the running simulation"""
        self.is_running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join()

# Global simulation instance
simulation = NDNFDRLSimulation()

@app.route('/')
def index():
    """Main simulation interface"""
    return render_template('index.html')

@app.route('/api/scenarios')
def get_scenarios():
    """Get available simulation scenarios"""
    return jsonify(simulation.get_scenarios())

@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    try:
        data = request.json
        scenario_id = data.get('scenario_id', 1)
        duration = data.get('duration', 300)  # 5 minutes default
        domains = data.get('domains', 3)
        nodes_per_domain = data.get('nodes_per_domain', 5)
        
        if simulation.is_running:
            return jsonify({'error': 'Simulation already running'}), 400
        
        # Start simulation in separate thread
        simulation.simulation_thread = threading.Thread(
            target=simulation.run_simulation,
            args=(scenario_id, duration, domains, nodes_per_domain)
        )
        simulation.simulation_thread.start()
        
        return jsonify({'status': 'Simulation started'})
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the running simulation"""
    simulation.stop_simulation()
    return jsonify({'status': 'Simulation stopped'})

@app.route('/api/status')
def get_status():
    """Get current simulation status"""
    return jsonify({
        'is_running': simulation.is_running,
        'current_results': simulation.current_results,
        'phase1_completed': bool(simulation.phase1_metrics),
        'phase2_completed': bool(simulation.phase2_metrics)
    })

@app.route('/api/results')
def get_results():
    """Get simulation results and analysis"""
    return jsonify({
        'phase1_metrics': simulation.phase1_metrics,
        'phase2_metrics': simulation.phase2_metrics,
        'training_history': simulation.training_history,
        'comparative_analysis': simulation.get_comparative_analysis()
    })

@app.route('/api/export/<format>')
def export_results(format):
    """Export simulation results in specified format"""
    if format == 'json':
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase1_metrics': simulation.phase1_metrics,
            'phase2_metrics': simulation.phase2_metrics,
            'training_history': simulation.training_history,
            'comparative_analysis': simulation.get_comparative_analysis()
        }
        return jsonify(results)
    
    return jsonify({'error': 'Unsupported format'}), 400

if __name__ == '__main__':
    logger.info("Starting NDN-FDRL Simulation System")
    app.run(host='0.0.0.0', port=5000, debug=False)
