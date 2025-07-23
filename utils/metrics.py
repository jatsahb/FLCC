"""
Metrics Collection and Analysis for NDN-FDRL System
Comprehensive metrics collection, analysis, and reporting for simulation results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import statistics

class MetricsCollector:
    """Comprehensive metrics collection and analysis for NDN-FDRL simulation"""
    
    def __init__(self):
        # Core metrics storage
        self.network_metrics = defaultdict(list)
        self.agent_metrics = defaultdict(lambda: defaultdict(list))
        self.federated_metrics = defaultdict(list)
        self.stakeholder_metrics = defaultdict(list)
        self.scenario_metrics = defaultdict(list)
        
        # Time series data
        self.timestamps = []
        self.phase_indicators = []  # 1 for baseline, 2 for FL
        
        # Performance tracking
        self.phase1_data = defaultdict(list)
        self.phase2_data = defaultdict(list)
        
        # Statistical analysis
        self.statistical_tests = {}
        self.correlation_analysis = {}
        
        # Real-time monitoring
        self.monitoring_window = deque(maxlen=100)
        self.alert_thresholds = self._set_default_thresholds()
        
        self.logger = logging.getLogger("MetricsCollector")
        
    def _set_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Set default alert thresholds for metrics"""
        return {
            'congestion_level': {'high': 0.8, 'critical': 0.9},
            'packet_loss_rate': {'high': 0.1, 'critical': 0.2},
            'latency': {'high': 200, 'critical': 500},
            'cache_hit_rate': {'low': 0.3, 'critical': 0.2},
            'consumer_satisfaction': {'low': 0.6, 'critical': 0.4}
        }
        
    def record_network_metrics(self, metrics: Dict[str, float], timestamp: datetime, phase: int):
        """Record network-level metrics"""
        self.timestamps.append(timestamp)
        self.phase_indicators.append(phase)
        
        for metric_name, value in metrics.items():
            self.network_metrics[metric_name].append(value)
            
            # Store in phase-specific data
            if phase == 1:
                self.phase1_data[metric_name].append(value)
            else:
                self.phase2_data[metric_name].append(value)
                
        # Update monitoring window
        self.monitoring_window.append({
            'timestamp': timestamp,
            'phase': phase,
            'metrics': metrics.copy()
        })
        
        # Check for alerts
        self._check_alerts(metrics)
        
    def record_agent_metrics(self, agent_id: int, metrics: Dict[str, float], 
                           timestamp: datetime, phase: int):
        """Record agent-specific metrics"""
        for metric_name, value in metrics.items():
            self.agent_metrics[agent_id][metric_name].append(value)
            
    def record_federated_metrics(self, metrics: Dict[str, Any], timestamp: datetime):
        """Record federated learning metrics"""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.federated_metrics[metric_name].append(value)
                
    def record_stakeholder_metrics(self, metrics: Dict[str, Any], timestamp: datetime, phase: int):
        """Record stakeholder (consumer/producer/router) metrics"""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.stakeholder_metrics[metric_name].append(value)
                
    def record_scenario_metrics(self, scenario_id: int, metrics: Dict[str, float], 
                              timestamp: datetime):
        """Record scenario-specific metrics"""
        scenario_data = {
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'metrics': metrics.copy()
        }
        self.scenario_metrics['scenario_data'].append(scenario_data)
        
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check metrics against alert thresholds"""
        for metric_name, value in metrics.items():
            if metric_name in self.alert_thresholds:
                thresholds = self.alert_thresholds[metric_name]
                
                if 'critical' in thresholds and value >= thresholds['critical']:
                    self.logger.warning(f"CRITICAL: {metric_name} = {value:.4f} "
                                      f"(threshold: {thresholds['critical']})")
                elif 'high' in thresholds and value >= thresholds['high']:
                    self.logger.warning(f"HIGH: {metric_name} = {value:.4f} "
                                      f"(threshold: {thresholds['high']})")
                elif 'low' in thresholds and value <= thresholds['low']:
                    self.logger.warning(f"LOW: {metric_name} = {value:.4f} "
                                      f"(threshold: {thresholds['low']})")
                    
    def calculate_statistical_summary(self, data: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistical summary"""
        if not data:
            return {}
            
        data_array = np.array(data)
        
        return {
            'count': len(data),
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'var': float(np.var(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'q25': float(np.percentile(data_array, 25)),
            'q75': float(np.percentile(data_array, 75)),
            'iqr': float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
            'skewness': float(self._calculate_skewness(data_array)),
            'kurtosis': float(self._calculate_kurtosis(data_array))
        }
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
        
    def compare_phases(self) -> Dict[str, Dict[str, float]]:
        """Compare Phase 1 (baseline) vs Phase 2 (federated learning)"""
        comparison = {}
        
        # Get common metrics between phases
        common_metrics = set(self.phase1_data.keys()) & set(self.phase2_data.keys())
        
        for metric in common_metrics:
            phase1_data = self.phase1_data[metric]
            phase2_data = self.phase2_data[metric]
            
            if not phase1_data or not phase2_data:
                continue
                
            phase1_stats = self.calculate_statistical_summary(phase1_data)
            phase2_stats = self.calculate_statistical_summary(phase2_data)
            
            # Calculate improvement metrics
            mean_improvement = ((phase2_stats['mean'] - phase1_stats['mean']) / 
                              abs(phase1_stats['mean'])) * 100 if phase1_stats['mean'] != 0 else 0
            
            # For negative metrics (latency, loss), improvement is when phase2 < phase1
            is_negative_metric = metric in ['latency', 'packet_loss_rate', 'congestion_level']
            improved = (mean_improvement < 0) if is_negative_metric else (mean_improvement > 0)
            
            comparison[metric] = {
                'phase1_stats': phase1_stats,
                'phase2_stats': phase2_stats,
                'improvement_percent': abs(mean_improvement),
                'improved': improved,
                'effect_size': self._calculate_effect_size(phase1_data, phase2_data),
                'statistical_significance': self._test_statistical_significance(phase1_data, phase2_data)
            }
            
        return comparison
        
    def _calculate_effect_size(self, data1: List[float], data2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if not data1 or not data2:
            return 0.0
            
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
            
        return abs(mean1 - mean2) / pooled_std
        
    def _test_statistical_significance(self, data1: List[float], data2: List[float]) -> Dict[str, float]:
        """Perform statistical significance tests"""
        if len(data1) < 3 or len(data2) < 3:
            return {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False}
            
        # Perform two-sample t-test (assuming unequal variances)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # Standard error
        se = np.sqrt(var1/n1 + var2/n2)
        
        if se == 0:
            return {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False}
            
        # T-statistic
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom (Welch's formula)
        df = ((var1/n1 + var2/n2) ** 2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Approximate p-value (simplified)
        # In practice, use scipy.stats.ttest_ind
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'degrees_of_freedom': float(df),
            'significant': p_value < 0.05
        }
        
    def _t_cdf(self, t: float, df: float) -> float:
        """Simplified t-distribution CDF approximation"""
        # Very basic approximation - in practice use scipy.stats
        if df > 30:
            # Use normal approximation for large df
            return 0.5 * (1 + np.sign(t) * np.sqrt(1 - np.exp(-2 * t**2 / np.pi)))
        else:
            # Basic approximation for smaller df
            return 0.5 + (t / (2 * np.sqrt(df))) * (1 - t**2 / (4 * df))
            
    def calculate_correlation_matrix(self, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for specified metrics"""
        correlation_matrix = {}
        
        for metric1 in metrics:
            correlation_matrix[metric1] = {}
            data1 = self.network_metrics.get(metric1, [])
            
            for metric2 in metrics:
                data2 = self.network_metrics.get(metric2, [])
                
                if len(data1) >= 2 and len(data2) >= 2 and len(data1) == len(data2):
                    corr = np.corrcoef(data1, data2)[0, 1]
                    correlation_matrix[metric1][metric2] = float(corr) if not np.isnan(corr) else 0.0
                else:
                    correlation_matrix[metric1][metric2] = 0.0
                    
        return correlation_matrix
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'summary': {
                'total_duration': len(self.timestamps),
                'phase1_duration': len(self.phase1_data.get('average_latency', [])),
                'phase2_duration': len(self.phase2_data.get('average_latency', [])),
                'metrics_collected': list(self.network_metrics.keys()),
                'total_agents': len(self.agent_metrics),
                'federated_rounds': len(self.federated_metrics.get('round_number', []))
            },
            'network_performance': {},
            'phase_comparison': self.compare_phases(),
            'agent_performance': {},
            'federated_learning': {},
            'stakeholder_analysis': {},
            'statistical_analysis': {},
            'recommendations': self._generate_recommendations()
        }
        
        # Network performance summary
        for metric_name, data in self.network_metrics.items():
            report['network_performance'][metric_name] = self.calculate_statistical_summary(data)
            
        # Agent performance summary
        for agent_id, agent_data in self.agent_metrics.items():
            report['agent_performance'][agent_id] = {}
            for metric_name, data in agent_data.items():
                report['agent_performance'][agent_id][metric_name] = self.calculate_statistical_summary(data)
                
        # Federated learning summary
        for metric_name, data in self.federated_metrics.items():
            if isinstance(data[0] if data else 0, (int, float)):
                report['federated_learning'][metric_name] = self.calculate_statistical_summary(data)
                
        # Stakeholder analysis
        for metric_name, data in self.stakeholder_metrics.items():
            if isinstance(data[0] if data else 0, (int, float)):
                report['stakeholder_analysis'][metric_name] = self.calculate_statistical_summary(data)
                
        # Statistical analysis
        key_metrics = ['average_latency', 'packet_loss_rate', 'cache_hit_rate', 'congestion_level']
        available_metrics = [m for m in key_metrics if m in self.network_metrics]
        
        if available_metrics:
            report['statistical_analysis']['correlation_matrix'] = self.calculate_correlation_matrix(available_metrics)
            
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on collected data"""
        recommendations = []
        
        # Analyze phase comparison
        phase_comparison = self.compare_phases()
        
        for metric, comparison in phase_comparison.items():
            if comparison['improved']:
                effect_size = comparison['effect_size']
                if effect_size > 0.8:
                    recommendations.append(
                        f"Federated learning shows large improvement in {metric} "
                        f"({comparison['improvement_percent']:.1f}% improvement)"
                    )
                elif effect_size > 0.5:
                    recommendations.append(
                        f"Federated learning shows moderate improvement in {metric} "
                        f"({comparison['improvement_percent']:.1f}% improvement)"
                    )
            else:
                recommendations.append(
                    f"Consider tuning federated learning parameters for {metric} "
                    f"(currently {comparison['improvement_percent']:.1f}% worse)"
                )
                
        # Check for high congestion
        if 'congestion_level' in self.network_metrics:
            avg_congestion = np.mean(self.network_metrics['congestion_level'])
            if avg_congestion > 0.7:
                recommendations.append(
                    "High average congestion detected. Consider increasing cache sizes "
                    "or improving load balancing algorithms."
                )
                
        # Check cache efficiency
        if 'cache_hit_rate' in self.network_metrics:
            avg_cache_hit = np.mean(self.network_metrics['cache_hit_rate'])
            if avg_cache_hit < 0.5:
                recommendations.append(
                    "Low cache hit rate detected. Consider optimizing cache policies "
                    "or content placement strategies."
                )
                
        # Check packet loss
        if 'packet_loss_rate' in self.network_metrics:
            avg_loss = np.mean(self.network_metrics['packet_loss_rate'])
            if avg_loss > 0.05:
                recommendations.append(
                    "High packet loss rate detected. Consider implementing "
                    "adaptive congestion control mechanisms."
                )
                
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges.")
            
        return recommendations
        
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export metrics to pandas DataFrame for analysis"""
        # Prepare data for DataFrame
        data = []
        
        for i, timestamp in enumerate(self.timestamps):
            row = {
                'timestamp': timestamp,
                'phase': self.phase_indicators[i]
            }
            
            # Add network metrics
            for metric_name, values in self.network_metrics.items():
                if i < len(values):
                    row[f'network_{metric_name}'] = values[i]
                    
            data.append(row)
            
        return pd.DataFrame(data)
        
    def export_to_json(self, filepath: str):
        """Export metrics to JSON file"""
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_duration': len(self.timestamps),
                'metrics_collected': list(self.network_metrics.keys())
            },
            'network_metrics': {k: v for k, v in self.network_metrics.items()},
            'agent_metrics': {str(k): v for k, v in self.agent_metrics.items()},
            'federated_metrics': {k: v for k, v in self.federated_metrics.items()},
            'stakeholder_metrics': {k: v for k, v in self.stakeholder_metrics.items()},
            'timestamps': [t.isoformat() for t in self.timestamps],
            'phase_indicators': self.phase_indicators,
            'performance_report': self.generate_performance_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
    def get_real_time_summary(self) -> Dict[str, Any]:
        """Get real-time metrics summary for dashboard"""
        if not self.monitoring_window:
            return {}
            
        recent_data = list(self.monitoring_window)[-10:]  # Last 10 data points
        
        summary = {
            'current_phase': recent_data[-1]['phase'] if recent_data else 1,
            'latest_timestamp': recent_data[-1]['timestamp'].isoformat() if recent_data else None,
            'metrics_trend': {},
            'alerts': [],
            'performance_indicators': {}
        }
        
        # Calculate trends for key metrics
        key_metrics = ['average_latency', 'packet_loss_rate', 'cache_hit_rate', 'congestion_level']
        
        for metric in key_metrics:
            values = [data['metrics'].get(metric, 0) for data in recent_data if metric in data['metrics']]
            if len(values) >= 2:
                trend = 'increasing' if values[-1] > values[0] else 'decreasing'
                change_percent = ((values[-1] - values[0]) / abs(values[0])) * 100 if values[0] != 0 else 0
                
                summary['metrics_trend'][metric] = {
                    'trend': trend,
                    'change_percent': change_percent,
                    'current_value': values[-1]
                }
                
        return summary
        
    def reset(self):
        """Reset all collected metrics"""
        self.network_metrics.clear()
        self.agent_metrics.clear()
        self.federated_metrics.clear()
        self.stakeholder_metrics.clear()
        self.scenario_metrics.clear()
        
        self.timestamps.clear()
        self.phase_indicators.clear()
        self.phase1_data.clear()
        self.phase2_data.clear()
        
        self.statistical_tests.clear()
        self.correlation_analysis.clear()
        self.monitoring_window.clear()
        
        self.logger.info("Metrics collector reset completed")
