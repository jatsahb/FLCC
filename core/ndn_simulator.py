"""
NDN Network Simulator
Comprehensive simulation of Named Data Networking with realistic traffic patterns and behaviors
"""

import numpy as np
import random
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
import json
import logging

class NDNPacket:
    """NDN packet structure"""
    def __init__(self, name: str, packet_type: str, timestamp: float, size: int = 1024):
        self.name = name
        self.packet_type = packet_type  # 'interest', 'data', 'nack'
        self.timestamp = timestamp
        self.size = size
        self.hop_count = 0
        self.faces = []
        
class PIT:
    """Pending Interest Table"""
    def __init__(self, max_entries: int = 10000):
        self.entries = {}  # name -> {faces, timestamp, nonce}
        self.max_entries = max_entries
        
    def add_entry(self, name: str, face_id: int, nonce: str):
        """Add PIT entry"""
        if len(self.entries) >= self.max_entries:
            # Remove oldest entry
            oldest = min(self.entries.keys(), key=lambda k: self.entries[k]['timestamp'])
            del self.entries[oldest]
            
        if name not in self.entries:
            self.entries[name] = {'faces': set(), 'timestamp': 0, 'nonces': set()}
        
        self.entries[name]['faces'].add(face_id)
        self.entries[name]['nonces'].add(nonce)
        self.entries[name]['timestamp'] = max(self.entries[name]['timestamp'], time.time())
        
    def remove_entry(self, name: str):
        """Remove PIT entry"""
        if name in self.entries:
            del self.entries[name]
            
    def get_faces(self, name: str) -> set:
        """Get faces for content name"""
        return self.entries.get(name, {}).get('faces', set())
        
    def get_entry_count(self) -> int:
        """Get current PIT entry count"""
        return len(self.entries)

class FIB:
    """Forwarding Information Base"""
    def __init__(self):
        self.entries = {}  # prefix -> {face_id: cost}
        
    def add_route(self, prefix: str, face_id: int, cost: int = 1):
        """Add FIB route"""
        if prefix not in self.entries:
            self.entries[prefix] = {}
        self.entries[prefix][face_id] = cost
        
    def lookup(self, name: str) -> List[Tuple[int, int]]:
        """Longest prefix match lookup"""
        best_match = ""
        for prefix in self.entries.keys():
            if name.startswith(prefix) and len(prefix) > len(best_match):
                best_match = prefix
                
        if best_match:
            return [(face_id, cost) for face_id, cost in self.entries[best_match].items()]
        return []

class CS:
    """Content Store (Cache)"""
    def __init__(self, max_size: int = 1000):
        self.entries = {}  # name -> (data, timestamp, access_count)
        self.max_size = max_size
        self.access_order = deque()
        
    def add_content(self, name: str, data: bytes):
        """Add content with LRU eviction"""
        if len(self.entries) >= self.max_size:
            # LRU eviction
            while self.access_order and self.access_order[0] not in self.entries:
                self.access_order.popleft()
            if self.access_order:
                oldest = self.access_order.popleft()
                if oldest in self.entries:
                    del self.entries[oldest]
                    
        self.entries[name] = (data, time.time(), 0)
        self.access_order.append(name)
        
    def get_content(self, name: str) -> Optional[bytes]:
        """Get content and update access"""
        if name in self.entries:
            data, timestamp, access_count = self.entries[name]
            self.entries[name] = (data, timestamp, access_count + 1)
            # Move to end for LRU
            try:
                self.access_order.remove(name)
                self.access_order.append(name)
            except ValueError:
                pass
            return data
        return None
        
    def get_utilization(self) -> float:
        """Get cache utilization ratio"""
        return len(self.entries) / self.max_size

class NDNNode:
    """NDN network node with complete NDN functionality"""
    def __init__(self, node_id: int, node_type: str = "router"):
        self.node_id = node_id
        self.node_type = node_type  # router, consumer, producer
        self.pit = PIT()
        self.fib = FIB()
        self.cs = CS()
        
        # Network metrics
        self.bandwidth = 10.0  # Mbps
        self.latency = 50.0    # ms
        self.queue = deque(maxlen=1000)
        self.packet_loss_rate = 0.0
        self.congestion_level = 0.0
        
        # Performance tracking
        self.interest_rate = 0.0
        self.data_rate = 0.0
        self.cache_hit_rate = 0.0
        self.queue_occupancy = 0.0
        
        # Statistics
        self.stats = {
            'interests_processed': 0,
            'data_forwarded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'packets_dropped': 0
        }
        
    def process_interest(self, interest: NDNPacket, incoming_face: int) -> List[NDNPacket]:
        """Process incoming Interest packet"""
        responses = []
        
        # Check CS first
        cached_data = self.cs.get_content(interest.name)
        if cached_data:
            # Cache hit - return data
            data_packet = NDNPacket(interest.name, 'data', time.time())
            responses.append(data_packet)
            self.stats['cache_hits'] += 1
            self.stats['data_forwarded'] += 1
            return responses
            
        self.stats['cache_misses'] += 1
        
        # Add to PIT
        nonce = f"nonce_{random.randint(1000, 9999)}"
        self.pit.add_entry(interest.name, incoming_face, nonce)
        
        # Check for duplicate interest
        if len(self.pit.get_faces(interest.name)) > 1:
            # Aggregate interests
            return responses
            
        # Forward interest
        routes = self.fib.lookup(interest.name)
        if routes:
            # Forward to best face (lowest cost)
            best_face = min(routes, key=lambda x: x[1])[0]
            if best_face != incoming_face:  # Avoid forwarding back
                forwarded_interest = NDNPacket(interest.name, 'interest', time.time())
                forwarded_interest.hop_count = interest.hop_count + 1
                responses.append(forwarded_interest)
                
        self.stats['interests_processed'] += 1
        return responses
        
    def process_data(self, data: NDNPacket) -> List[NDNPacket]:
        """Process incoming Data packet"""
        responses = []
        
        # Remove from PIT and get faces
        faces = self.pit.get_faces(data.name)
        self.pit.remove_entry(data.name)
        
        # Cache the data
        self.cs.add_content(data.name, b"data_content")
        
        # Forward to all faces in PIT
        for face_id in faces:
            forwarded_data = NDNPacket(data.name, 'data', time.time())
            responses.append(forwarded_data)
            
        self.stats['data_forwarded'] += len(responses)
        return responses
        
    def update_metrics(self, fl_enhanced: bool = False):
        """Update node performance metrics"""
        # Calculate rates
        current_time = time.time()
        time_window = 1.0  # 1 second window
        
        # Update congestion metrics
        self.queue_occupancy = len(self.queue) / self.queue.maxlen
        self.congestion_level = min(1.0, self.queue_occupancy + self.packet_loss_rate)
        
        # Apply FL enhancement effects
        if fl_enhanced:
            # FL improves network performance
            congestion_reduction = 0.8
            latency_improvement = 0.9
            loss_improvement = 0.7
        else:
            # Baseline performance
            congestion_reduction = 1.0
            latency_improvement = 1.0 
            loss_improvement = 1.0
        
        # Update congestion with FL effects
        self.congestion_level *= congestion_reduction
        
        # Update cache hit rate
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_requests > 0:
            self.cache_hit_rate = self.stats['cache_hits'] / total_requests
            
        # Simulate packet processing delay with FL improvements
        base_latency = 50
        if self.congestion_level > 0.7:
            self.latency = (base_latency + 200 * self.congestion_level) * latency_improvement
            self.packet_loss_rate = min(0.1, 0.05 * self.congestion_level ** 2) * loss_improvement
        else:
            self.latency = base_latency * latency_improvement
            self.packet_loss_rate *= loss_improvement
            
    def get_state(self) -> np.ndarray:
        """Get current node state for RL agent"""
        return np.array([
            self.congestion_level,
            self.queue_occupancy, 
            self.cache_hit_rate,
            self.packet_loss_rate,
            self.latency / 1000.0,  # Normalize to seconds
            self.bandwidth / 20.0,  # Normalize to max bandwidth
            self.interest_rate / 100.0,  # Normalize
            self.data_rate / 100.0   # Normalize
        ])

class NDNNetworkSimulator:
    """Complete NDN network simulator with advanced features"""
    def __init__(self, domains: int = 3, nodes_per_domain: int = 5):
        self.domains = domains
        self.nodes_per_domain = nodes_per_domain
        self.total_nodes = domains * nodes_per_domain
        
        # Initialize nodes
        self.nodes = {}
        self.initialize_network()
        
        # Traffic patterns
        self.content_popularity = self.generate_content_popularity()
        self.traffic_matrix = np.zeros((self.total_nodes, self.total_nodes))
        
        # Active scenario
        self.active_scenario = None
        self.scenario_start_time = None
        
        # Network state tracking
        self.network_metrics = {
            'average_latency': 0.0,
            'packet_loss_rate': 0.0,
            'cache_hit_rate': 0.0,
            'throughput': 0.0,
            'congestion_level': 0.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize_network(self):
        """Initialize network topology and nodes"""
        # Create nodes
        for i in range(self.total_nodes):
            domain = i // self.nodes_per_domain
            position_in_domain = i % self.nodes_per_domain
            
            # Assign node types
            if position_in_domain == 0:
                node_type = "producer"
            elif position_in_domain < 3:
                node_type = "router"
            else:
                node_type = "consumer"
                
            self.nodes[i] = NDNNode(i, node_type)
            
        # Configure FIB entries (simple routing)
        self.configure_routing()
        
        # Initialize traffic patterns
        self.initialize_traffic_patterns()
        
    def configure_routing(self):
        """Configure FIB entries for all nodes"""
        for node_id, node in self.nodes.items():
            domain = node_id // self.nodes_per_domain
            
            # Add routes to content prefixes
            for content_domain in range(self.domains):
                prefix = f"/domain{content_domain}"
                
                if domain == content_domain:
                    # Local domain - direct route to producer
                    producer_id = content_domain * self.nodes_per_domain
                    if node_id != producer_id:
                        node.fib.add_route(prefix, producer_id, 1)
                else:
                    # Remote domain - route through inter-domain link
                    gateway_id = domain * self.nodes_per_domain + 1  # Router
                    node.fib.add_route(prefix, gateway_id, 2)
                    
    def initialize_traffic_patterns(self):
        """Initialize realistic traffic patterns"""
        # Create content popularity distribution (Zipf)
        self.content_catalog = {}
        for domain in range(self.domains):
            for content_id in range(100):  # 100 contents per domain
                name = f"/domain{domain}/content{content_id}"
                popularity = 1.0 / ((content_id + 1) ** 0.8)  # Zipf distribution
                self.content_catalog[name] = popularity
                
    def generate_content_popularity(self) -> Dict[str, float]:
        """Generate Zipf-distributed content popularity"""
        popularity = {}
        for domain in range(self.domains):
            for i in range(100):
                content_name = f"/domain{domain}/content{i}"
                # Zipf distribution with parameter α = 0.8
                popularity[content_name] = 1.0 / ((i + 1) ** 0.8)
        return popularity
        
    def set_active_scenario(self, scenario: Dict):
        """Set the active congestion scenario"""
        self.active_scenario = scenario
        self.scenario_start_time = time.time()
        self.logger.info(f"Activated scenario: {scenario.get('name', 'Unknown')}")
        
    def apply_scenario_effects(self, fl_enhanced: bool = False):
        """Apply active scenario effects to network"""
        if not self.active_scenario:
            return
            
        scenario_type = self.active_scenario.get('type', '')
        severity = self.active_scenario.get('severity', 0.5)
        
        # Calculate scenario impact (reduced if FL is active)
        impact_factor = severity * (0.6 if fl_enhanced else 1.0)
        
        if scenario_type == 'demand_surge':
            # Increase request rates and cache misses
            for node in self.nodes.values():
                if node.node_type == 'router':
                    node.congestion_level = min(1.0, node.congestion_level + impact_factor * 0.3)
                    node.packet_loss_rate = min(0.2, node.packet_loss_rate + impact_factor * 0.1)
                    
        elif scenario_type == 'infrastructure_failure':
            # Increase latency and packet loss
            affected_nodes = random.sample(
                list(self.nodes.keys()), 
                max(1, int(len(self.nodes) * impact_factor * 0.3))
            )
            for node_id in affected_nodes:
                node = self.nodes[node_id]
                node.latency *= (1.0 + impact_factor * 2.0)
                node.packet_loss_rate = min(0.3, node.packet_loss_rate + impact_factor * 0.2)
                
        elif scenario_type == 'security_attack':
            # Increase processing overhead and PIT utilization
            for node in self.nodes.values():
                node.congestion_level = min(1.0, node.congestion_level + impact_factor * 0.4)
                
    def step(self, actions: Dict[int, float], timestep: int, fl_enhanced: bool = False) -> Dict:
        """Execute one simulation step"""
        # Apply agent actions to network
        self.apply_actions(actions)
        
        # Apply scenario effects if active
        self.apply_scenario_effects(fl_enhanced)
        
        # Generate traffic
        self.generate_traffic(timestep)
        
        # Process packets at all nodes
        self.process_network_traffic()
        
        # Update network state
        self.update_network_state(fl_enhanced)
        
        # Calculate step metrics
        step_metrics = self.calculate_step_metrics()
        
        return step_metrics
        
    def apply_actions(self, actions: Dict[int, float]):
        """Apply DDPG agent actions to network"""
        for node_id, action in actions.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Transform action [-1, 1] to bandwidth adjustment
                bandwidth_change = 0.2 * np.tanh(3 * action)
                new_bandwidth = node.bandwidth * (1 + bandwidth_change)
                
                # Apply constraints
                node.bandwidth = np.clip(new_bandwidth, 1.0, 20.0)
                
    def generate_traffic(self, timestep: int):
        """Generate realistic NDN traffic"""
        # Base request rate per node
        base_rate = 10.0  # requests per second
        
        for node_id, node in self.nodes.items():
            if node.node_type == 'consumer':
                # Generate interests based on content popularity
                request_rate = base_rate * (1.0 + 0.3 * random.random())
                
                if random.random() < request_rate / 100.0:  # Probability per step
                    # Select content based on popularity
                    content_names = list(self.content_catalog.keys())
                    weights = [self.content_catalog[name] for name in content_names]
                    content_name = random.choices(content_names, weights=weights)[0]
                    
                    # Create interest packet
                    interest = NDNPacket(content_name, 'interest', time.time())
                    node.queue.append(interest)
                    
    def process_network_traffic(self):
        """Process packets at all nodes"""
        for node in self.nodes.values():
            # Process queued packets
            packets_to_process = list(node.queue)
            node.queue.clear()
            
            for packet in packets_to_process:
                if packet.packet_type == 'interest':
                    responses = node.process_interest(packet, 0)
                    # Add responses to appropriate queues (simplified)
                elif packet.packet_type == 'data':
                    responses = node.process_data(packet)
                    # Add responses to appropriate queues (simplified)
                    
    def update_network_state(self, fl_enhanced: bool = False):
        """Update network state with FL enhancements"""
        total_latency = 0.0
        total_loss = 0.0
        total_cache_hits = 0
        total_cache_requests = 0
        total_congestion = 0.0
        
        for node in self.nodes.values():
            node.update_metrics(fl_enhanced)
            
            total_latency += node.latency
            total_loss += node.packet_loss_rate
            total_congestion += node.congestion_level
            
            total_cache_hits += node.stats['cache_hits']
            total_cache_requests += (node.stats['cache_hits'] + node.stats['cache_misses'])
            
        num_nodes = len(self.nodes)
        self.network_metrics = {
            'average_latency': total_latency / num_nodes,
            'packet_loss_rate': total_loss / num_nodes,
            'cache_hit_rate': total_cache_hits / max(1, total_cache_requests),
            'throughput': sum(node.bandwidth for node in self.nodes.values()) / num_nodes,
            'congestion_level': total_congestion / num_nodes
        }
        
    def calculate_step_metrics(self) -> Dict:
        """Calculate metrics for current simulation step"""
        return {
            'average_latency': self.network_metrics['average_latency'],
            'packet_loss_rate': self.network_metrics['packet_loss_rate'],
            'cache_hit_rate': self.network_metrics['cache_hit_rate'],
            'throughput': self.network_metrics['throughput'],
            'congestion_level': self.network_metrics['congestion_level'],
            'pit_utilization': np.mean([node.pit.get_entry_count() / node.pit.max_entries for node in self.nodes.values()]),
            'cache_utilization': np.mean([node.cs.get_utilization() for node in self.nodes.values()]),
            'total_interests': sum(node.stats['interests_processed'] for node in self.nodes.values()),
            'total_data_packets': sum(node.stats['data_forwarded'] for node in self.nodes.values())
        }
        
    def get_network_state(self, fl_enhanced: bool = False) -> Dict:
        """Get current network state"""
        return {
            'nodes': {node_id: {
                'type': node.node_type,
                'latency': node.latency,
                'congestion': node.congestion_level,
                'cache_hit_rate': node.cache_hit_rate,
                'queue_occupancy': node.queue_occupancy
            } for node_id, node in self.nodes.items()},
            'metrics': self.network_metrics,
            'fl_enhanced': fl_enhanced
        }
        
    def get_node_state(self, node_id: int) -> np.ndarray:
        """Get state for specific node"""
        if node_id in self.nodes:
            return self.nodes[node_id].get_state()
        else:
            return np.zeros(8)  # Default state
            
    def reset(self):
        """Reset simulator state"""
        for node in self.nodes.values():
            node.pit.entries.clear()
            node.cs.entries.clear()
            node.queue.clear()
            node.stats = {
                'interests_processed': 0,
                'data_forwarded': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'packets_dropped': 0
            }
            node.latency = 50.0
            node.bandwidth = 10.0
            node.packet_loss_rate = 0.0
            node.congestion_level = 0.0
            
        self.active_scenario = None
        self.scenario_start_time = None
        
        self.logger.info("NDN simulator reset completed")
