"""
Stakeholder Monitor for NDN Networks
Monitors and models behavior of consumers, producers, and routers in the NDN ecosystem
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import random
import logging
from datetime import datetime, timedelta

class Consumer:
    """NDN Consumer entity with behavior modeling"""
    def __init__(self, consumer_id: str, preferences: Dict = None):
        self.consumer_id = consumer_id
        self.preferences = preferences or self.generate_default_preferences()
        
        # Request patterns
        self.request_rate = self.preferences.get('request_rate', 10)  # requests/minute
        self.content_preferences = self.preferences.get('content_types', ['video', 'web', 'file'])
        self.tolerance_latency = self.preferences.get('latency_tolerance', 100)  # ms
        self.quality_expectation = self.preferences.get('quality_expectation', 0.8)
        
        # Behavior state
        self.satisfaction_score = 1.0
        self.current_requests = []
        self.completed_requests = deque(maxlen=100)
        self.total_requests = 0
        self.successful_requests = 0
        
        # Experience tracking
        self.latency_history = deque(maxlen=50)
        self.quality_history = deque(maxlen=50)
        self.cache_hit_ratio = 0.0
        
        # Adaptation parameters
        self.patience_level = random.uniform(0.3, 1.0)
        self.adaptation_rate = random.uniform(0.1, 0.3)
        
    def generate_default_preferences(self) -> Dict:
        """Generate realistic consumer preferences"""
        profiles = [
            {'type': 'casual', 'request_rate': 5, 'latency_tolerance': 200, 'quality_expectation': 0.6},
            {'type': 'power_user', 'request_rate': 20, 'latency_tolerance': 50, 'quality_expectation': 0.9},
            {'type': 'enterprise', 'request_rate': 15, 'latency_tolerance': 100, 'quality_expectation': 0.95},
            {'type': 'mobile', 'request_rate': 8, 'latency_tolerance': 300, 'quality_expectation': 0.5}
        ]
        
        profile = random.choice(profiles)
        return {
            'profile_type': profile['type'],
            'request_rate': profile['request_rate'],
            'latency_tolerance': profile['latency_tolerance'],
            'quality_expectation': profile['quality_expectation'],
            'content_types': random.sample(['video', 'web', 'file', 'stream', 'download'], k=3)
        }
        
    def generate_interest(self, content_catalog: Dict[str, float]) -> Optional[Dict]:
        """Generate interest based on preferences and content popularity"""
        if random.random() > (self.request_rate / 60):  # Convert to per-second probability
            return None
            
        # Select content based on preferences and popularity
        preferred_contents = [
            name for name in content_catalog.keys()
            if any(pref in name for pref in self.content_preferences)
        ]
        
        if not preferred_contents:
            preferred_contents = list(content_catalog.keys())
            
        # Weighted selection based on popularity
        weights = [content_catalog[name] for name in preferred_contents]
        content_name = random.choices(preferred_contents, weights=weights)[0]
        
        interest = {
            'consumer_id': self.consumer_id,
            'content_name': content_name,
            'timestamp': datetime.now(),
            'expected_quality': self.quality_expectation,
            'max_latency': self.tolerance_latency
        }
        
        self.current_requests.append(interest)
        self.total_requests += 1
        
        return interest
        
    def receive_data(self, data_packet: Dict):
        """Process received data packet and update satisfaction"""
        # Find corresponding request
        request = None
        for req in self.current_requests:
            if req['content_name'] == data_packet['content_name']:
                request = req
                self.current_requests.remove(req)
                break
                
        if not request:
            return  # Unsolicited data
            
        # Calculate response metrics
        response_time = (datetime.now() - request['timestamp']).total_seconds() * 1000  # ms
        quality_received = data_packet.get('quality', 0.5)
        
        # Update history
        self.latency_history.append(response_time)
        self.quality_history.append(quality_received)
        
        # Calculate satisfaction for this request
        latency_satisfaction = 1.0 if response_time <= request['max_latency'] else \
                             max(0.0, 1.0 - (response_time - request['max_latency']) / request['max_latency'])
        
        quality_satisfaction = min(1.0, quality_received / request['expected_quality'])
        
        request_satisfaction = 0.6 * latency_satisfaction + 0.4 * quality_satisfaction
        
        # Update overall satisfaction with adaptation
        self.satisfaction_score = (
            (1 - self.adaptation_rate) * self.satisfaction_score +
            self.adaptation_rate * request_satisfaction
        )
        
        # Store completed request
        completed_request = {
            'request': request,
            'response_time': response_time,
            'quality': quality_received,
            'satisfaction': request_satisfaction,
            'cache_hit': data_packet.get('from_cache', False)
        }
        self.completed_requests.append(completed_request)
        self.successful_requests += 1
        
    def timeout_requests(self, timeout_ms: int = 5000):
        """Handle request timeouts"""
        current_time = datetime.now()
        timed_out = []
        
        for request in self.current_requests[:]:
            if (current_time - request['timestamp']).total_seconds() * 1000 > timeout_ms:
                timed_out.append(request)
                self.current_requests.remove(request)
                
        # Penalize satisfaction for timeouts
        for _ in timed_out:
            self.satisfaction_score = (
                (1 - self.adaptation_rate) * self.satisfaction_score +
                self.adaptation_rate * 0.0  # Zero satisfaction for timeout
            )
            
    def get_metrics(self) -> Dict:
        """Get consumer performance metrics"""
        return {
            'consumer_id': self.consumer_id,
            'satisfaction_score': self.satisfaction_score,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate': self.successful_requests / max(1, self.total_requests),
            'average_latency': np.mean(self.latency_history) if self.latency_history else 0,
            'average_quality': np.mean(self.quality_history) if self.quality_history else 0,
            'pending_requests': len(self.current_requests),
            'profile_type': self.preferences.get('profile_type', 'unknown')
        }

class Producer:
    """NDN Producer entity with content generation"""
    def __init__(self, producer_id: str, content_domains: List[str]):
        self.producer_id = producer_id
        self.content_domains = content_domains
        
        # Content catalog
        self.content_catalog = self.generate_content_catalog()
        
        # Production metrics
        self.total_content_served = 0
        self.cache_efficiency = 0.0
        self.load_balance = 0.5  # 0 = underloaded, 1 = overloaded
        
        # Quality control
        self.content_quality = {}
        self.popularity_tracking = defaultdict(int)
        
        # Performance tracking
        self.response_times = deque(maxlen=100)
        self.served_content_history = deque(maxlen=500)
        
    def generate_content_catalog(self) -> Dict[str, Dict]:
        """Generate producer content catalog"""
        catalog = {}
        
        for domain in self.content_domains:
            for i in range(100):  # 100 pieces of content per domain
                content_name = f"/{domain}/content{i}"
                catalog[content_name] = {
                    'size': random.randint(1024, 10240),  # KB
                    'quality': random.uniform(0.5, 1.0),
                    'generation_cost': random.uniform(0.1, 1.0),
                    'cache_duration': random.randint(300, 3600),  # seconds
                    'access_count': 0
                }
                
        return catalog
        
    def serve_content(self, interest: Dict) -> Optional[Dict]:
        """Serve content for given interest"""
        content_name = interest['content_name']
        
        if content_name not in self.content_catalog:
            return None  # Content not available
            
        content_info = self.content_catalog[content_name]
        
        # Simulate processing time based on content size and current load
        processing_time = (content_info['size'] / 1024) * (1 + self.load_balance) * 10  # ms
        
        # Create data packet
        data_packet = {
            'content_name': content_name,
            'size': content_info['size'],
            'quality': content_info['quality'],
            'producer_id': self.producer_id,
            'processing_time': processing_time,
            'timestamp': datetime.now(),
            'from_cache': False
        }
        
        # Update metrics
        self.total_content_served += 1
        content_info['access_count'] += 1
        self.popularity_tracking[content_name] += 1
        self.response_times.append(processing_time)
        self.served_content_history.append(content_name)
        
        # Update load balance based on recent activity
        recent_requests = len([
            req for req in self.served_content_history
            if req  # Simple count of recent activity
        ])
        self.load_balance = min(1.0, recent_requests / 100)
        
        return data_packet
        
    def get_metrics(self) -> Dict:
        """Get producer performance metrics"""
        popular_content = max(self.popularity_tracking.items(), 
                            key=lambda x: x[1], default=('none', 0))
        
        return {
            'producer_id': self.producer_id,
            'total_content_served': self.total_content_served,
            'content_domains': self.content_domains,
            'catalog_size': len(self.content_catalog),
            'load_balance': self.load_balance,
            'average_response_time': np.mean(self.response_times) if self.response_times else 0,
            'most_popular_content': popular_content[0],
            'max_popularity_count': popular_content[1],
            'unique_content_served': len(set(self.served_content_history))
        }

class Router:
    """NDN Router entity with forwarding and caching"""
    def __init__(self, router_id: str, cache_size: int = 1000):
        self.router_id = router_id
        self.cache_size = cache_size
        
        # Router state
        self.forwarding_table = {}
        self.content_cache = {}
        self.pending_interests = defaultdict(list)
        
        # Performance metrics
        self.packets_forwarded = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.congestion_events = 0
        self.queue_size = 0
        self.max_queue_size = 1000
        
        # Monitoring
        self.latency_measurements = deque(maxlen=100)
        self.throughput_measurements = deque(maxlen=100)
        self.congestion_history = deque(maxlen=50)
        
    def forward_interest(self, interest: Dict) -> bool:
        """Forward interest packet"""
        # Check cache first
        content_name = interest['content_name']
        
        if content_name in self.content_cache:
            # Cache hit
            self.cache_hits += 1
            cached_data = self.content_cache[content_name]
            cached_data['from_cache'] = True
            return cached_data
        else:
            # Cache miss - forward interest
            self.cache_misses += 1
            self.pending_interests[content_name].append(interest)
            self.packets_forwarded += 1
            
            # Simulate queue management
            if self.queue_size < self.max_queue_size:
                self.queue_size += 1
                return True
            else:
                self.congestion_events += 1
                return False
                
    def forward_data(self, data_packet: Dict):
        """Forward data packet and cache if beneficial"""
        content_name = data_packet['content_name']
        
        # Cache decision (simple LRU with popularity consideration)
        if len(self.content_cache) < self.cache_size:
            self.content_cache[content_name] = data_packet.copy()
        else:
            # Remove least recently used content
            if self.content_cache:
                oldest_content = min(self.content_cache.keys(), 
                                   key=lambda k: self.content_cache[k].get('timestamp', 0))
                del self.content_cache[oldest_content]
                self.content_cache[content_name] = data_packet.copy()
                
        # Forward to interested faces
        if content_name in self.pending_interests:
            self.pending_interests[content_name].clear()
            
        # Update queue
        if self.queue_size > 0:
            self.queue_size -= 1
            
    def get_metrics(self) -> Dict:
        """Get router performance metrics"""
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        congestion_level = self.queue_size / self.max_queue_size
        
        return {
            'router_id': self.router_id,
            'packets_forwarded': self.packets_forwarded,
            'cache_hit_rate': cache_hit_rate,
            'cache_utilization': len(self.content_cache) / self.cache_size,
            'congestion_level': congestion_level,
            'congestion_events': self.congestion_events,
            'queue_occupancy': self.queue_size / self.max_queue_size,
            'average_latency': np.mean(self.latency_measurements) if self.latency_measurements else 0
        }

class StakeholderMonitor:
    """Monitor and coordinate all stakeholders in the NDN network"""
    
    def __init__(self, num_consumers: int, num_producers: int, num_routers: int):
        self.consumers = {}
        self.producers = {}
        self.routers = {}
        
        # Initialize stakeholders
        self.initialize_consumers(num_consumers)
        self.initialize_producers(num_producers)
        self.initialize_routers(num_routers)
        
        # Monitoring state
        self.total_requests = 0
        self.total_responses = 0
        self.total_timeouts = 0
        
        # Performance tracking
        self.network_satisfaction = deque(maxlen=100)
        self.system_metrics = {
            'consumer_satisfaction': 0.0,
            'producer_load_balance': 0.0,
            'router_efficiency': 0.0,
            'overall_performance': 0.0
        }
        
        self.logger = logging.getLogger("StakeholderMonitor")
        
    def initialize_consumers(self, num_consumers: int):
        """Initialize consumer entities"""
        for i in range(num_consumers):
            consumer_id = f"consumer_{i}"
            self.consumers[consumer_id] = Consumer(consumer_id)
            
    def initialize_producers(self, num_producers: int):
        """Initialize producer entities"""
        for i in range(num_producers):
            producer_id = f"producer_{i}"
            domains = [f"domain{i}"]  # Each producer handles one domain
            self.producers[producer_id] = Producer(producer_id, domains)
            
    def initialize_routers(self, num_routers: int):
        """Initialize router entities"""
        for i in range(num_routers):
            router_id = f"router_{i}"
            self.routers[router_id] = Router(router_id)
            
    def update(self, network_state: Dict) -> Dict:
        """Update stakeholder behaviors and collect metrics"""
        # Update consumers
        self.update_consumers(network_state)
        
        # Update producers
        self.update_producers(network_state)
        
        # Update routers
        self.update_routers(network_state)
        
        # Calculate system metrics
        system_metrics = self.calculate_system_metrics()
        
        return system_metrics
        
    def update_consumers(self, network_state: Dict):
        """Update consumer behavior based on network state"""
        for consumer in self.consumers.values():
            # Handle timeouts
            consumer.timeout_requests()
            
            # Generate new interests based on content catalog
            content_catalog = self.get_content_catalog()
            interest = consumer.generate_interest(content_catalog)
            
            if interest:
                self.total_requests += 1
                
                # Simulate content retrieval (simplified)
                if random.random() < 0.8:  # 80% success rate
                    data_packet = self.simulate_data_response(interest)
                    consumer.receive_data(data_packet)
                    self.total_responses += 1
                else:
                    self.total_timeouts += 1
                    
    def update_producers(self, network_state: Dict):
        """Update producer state based on network load"""
        for producer in self.producers.values():
            # Simulate load balancing based on network congestion
            avg_congestion = np.mean([
                node.get('congestion_level', 0.2) 
                for node in network_state.get('nodes', {}).values()
            ])
            
            # Adjust producer load based on network congestion
            producer.load_balance = min(1.0, producer.load_balance + avg_congestion * 0.1)
            
    def update_routers(self, network_state: Dict):
        """Update router state based on network traffic"""
        for router_id, router in self.routers.items():
            # Update router metrics based on network state
            if router_id in network_state.get('nodes', {}):
                node_state = network_state['nodes'][router_id]
                
                # Update congestion metrics
                router.congestion_history.append(node_state.get('congestion', 0.2))
                
                # Update latency measurements
                router.latency_measurements.append(node_state.get('latency', 50))
                
    def get_content_catalog(self) -> Dict[str, float]:
        """Get combined content catalog from all producers"""
        catalog = {}
        
        for producer in self.producers.values():
            for content_name in producer.content_catalog.keys():
                # Simple popularity model based on access count
                access_count = producer.content_catalog[content_name]['access_count']
                popularity = 1.0 / (1.0 + np.exp(-0.1 * access_count))  # Sigmoid
                catalog[content_name] = popularity
                
        return catalog
        
    def simulate_data_response(self, interest: Dict) -> Dict:
        """Simulate data response for consumer interest"""
        content_name = interest['content_name']
        
        # Find appropriate producer
        for producer in self.producers.values():
            if content_name in producer.content_catalog:
                return producer.serve_content(interest)
                
        # Default response if no producer found
        return {
            'content_name': content_name,
            'size': 1024,
            'quality': 0.5,
            'producer_id': 'unknown',
            'processing_time': 100,
            'timestamp': datetime.now(),
            'from_cache': False
        }
        
    def calculate_system_metrics(self) -> Dict:
        """Calculate overall system performance metrics"""
        # Consumer satisfaction
        consumer_satisfactions = [
            consumer.satisfaction_score for consumer in self.consumers.values()
        ]
        avg_consumer_satisfaction = np.mean(consumer_satisfactions) if consumer_satisfactions else 0.0
        
        # Producer load balance
        producer_loads = [
            producer.load_balance for producer in self.producers.values()
        ]
        avg_producer_load = np.mean(producer_loads) if producer_loads else 0.5
        
        # Router efficiency
        router_efficiencies = []
        for router in self.routers.values():
            cache_hit_rate = router.cache_hits / max(1, router.cache_hits + router.cache_misses)
            congestion_penalty = 1.0 - (router.queue_size / router.max_queue_size)
            efficiency = cache_hit_rate * congestion_penalty
            router_efficiencies.append(efficiency)
            
        avg_router_efficiency = np.mean(router_efficiencies) if router_efficiencies else 0.5
        
        # Overall performance
        overall_performance = (
            0.4 * avg_consumer_satisfaction +
            0.3 * (1.0 - avg_producer_load) +  # Lower load is better
            0.3 * avg_router_efficiency
        )
        
        # Update system metrics
        self.system_metrics = {
            'consumer_satisfaction': avg_consumer_satisfaction,
            'producer_load_balance': avg_producer_load,
            'router_efficiency': avg_router_efficiency,
            'overall_performance': overall_performance,
            'total_requests': self.total_requests,
            'total_responses': self.total_responses,
            'success_rate': self.total_responses / max(1, self.total_requests)
        }
        
        return self.system_metrics
        
    def get_detailed_metrics(self) -> Dict:
        """Get detailed metrics for all stakeholders"""
        return {
            'consumers': {cid: consumer.get_metrics() for cid, consumer in self.consumers.items()},
            'producers': {pid: producer.get_metrics() for pid, producer in self.producers.items()},
            'routers': {rid: router.get_metrics() for rid, router in self.routers.items()},
            'system': self.system_metrics
        }
        
    def reset(self):
        """Reset all stakeholder states"""
        for consumer in self.consumers.values():
            consumer.satisfaction_score = 1.0
            consumer.current_requests.clear()
            consumer.completed_requests.clear()
            consumer.total_requests = 0
            consumer.successful_requests = 0
            consumer.latency_history.clear()
            consumer.quality_history.clear()
            
        for producer in self.producers.values():
            producer.total_content_served = 0
            producer.load_balance = 0.5
            producer.popularity_tracking.clear()
            producer.response_times.clear()
            producer.served_content_history.clear()
            
        for router in self.routers.values():
            router.packets_forwarded = 0
            router.cache_hits = 0
            router.cache_misses = 0
            router.congestion_events = 0
            router.queue_size = 0
            router.content_cache.clear()
            router.pending_interests.clear()
            router.latency_measurements.clear()
            router.throughput_measurements.clear()
            router.congestion_history.clear()
            
        self.total_requests = 0
        self.total_responses = 0
        self.total_timeouts = 0
        self.network_satisfaction.clear()
        
        self.logger.info("All stakeholders reset completed")
