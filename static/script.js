/**
 * NDN-FDRL Simulation System Frontend JavaScript
 * Handles user interactions, real-time updates, and data visualization
 */

class SimulationInterface {
    constructor() {
        this.isRunning = false;
        this.statusInterval = null;
        this.trainingChart = null;
        this.progressChart = null;
        this.scenarios = [];
        
        this.init();
    }
    
    init() {
        this.loadScenarios();
        this.setupEventListeners();
        this.initializeCharts();
    }
    
    async loadScenarios() {
        try {
            const response = await fetch('/api/scenarios');
            this.scenarios = await response.json();
            this.populateScenarioSelect();
        } catch (error) {
            console.error('Failed to load scenarios:', error);
            this.showError('Failed to load scenarios');
        }
    }
    
    populateScenarioSelect() {
        const select = document.getElementById('scenario-select');
        select.innerHTML = '<option value="">Select a scenario...</option>';
        
        this.scenarios.forEach(scenario => {
            const option = document.createElement('option');
            option.value = scenario.id;
            option.textContent = `${scenario.id}. ${scenario.name}`;
            select.appendChild(option);
        });
        
        // Add change event listener
        select.addEventListener('change', (e) => {
            this.showScenarioDescription(e.target.value);
        });
    }
    
    showScenarioDescription(scenarioId) {
        const scenario = this.scenarios.find(s => s.id == scenarioId);
        const descriptionDiv = document.getElementById('scenario-description');
        
        if (scenario) {
            document.getElementById('scenario-desc').textContent = scenario.description;
            document.getElementById('scenario-severity').textContent = scenario.severity;
            document.getElementById('scenario-severity').className = `severity-badge ${scenario.severity}`;
            document.getElementById('scenario-duration').textContent = scenario.duration;
            
            const effectsList = document.getElementById('scenario-effects-list');
            effectsList.innerHTML = '';
            scenario.primary_effects.forEach(effect => {
                const li = document.createElement('li');
                li.textContent = effect;
                effectsList.appendChild(li);
            });
            
            descriptionDiv.style.display = 'block';
            descriptionDiv.classList.add('fade-in');
        } else {
            descriptionDiv.style.display = 'none';
        }
    }
    
    setupEventListeners() {
        document.getElementById('start-btn').addEventListener('click', () => this.startSimulation());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopSimulation());
        document.getElementById('export-btn').addEventListener('click', () => this.exportResults());
    }
    
    initializeCharts() {
        // Training progress chart
        const trainingCtx = document.getElementById('training-chart');
        if (trainingCtx) {
            this.trainingChart = new Chart(trainingCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Model Accuracy',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y'
                    }, {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Federated Learning Progress'
                        }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            min: 0,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Accuracy'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: 0,
                            max: 2,
                            title: {
                                display: true,
                                text: 'Loss'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });
        }
    }
    
    async startSimulation() {
        const scenarioId = document.getElementById('scenario-select').value;
        const duration = parseInt(document.getElementById('duration-input').value);
        const domains = parseInt(document.getElementById('domains-input').value);
        const nodesPerDomain = parseInt(document.getElementById('nodes-input').value);
        
        if (!scenarioId) {
            this.showError('Please select a scenario');
            return;
        }
        
        if (duration < 60 || duration > 3600) {
            this.showError('Duration must be between 60 and 3600 seconds');
            return;
        }
        
        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    scenario_id: parseInt(scenarioId),
                    duration: duration,
                    domains: domains,
                    nodes_per_domain: nodesPerDomain
                })
            });
            
            if (response.ok) {
                this.isRunning = true;
                this.updateUIForRunningState();
                this.startStatusUpdates();
                this.showSuccess('Simulation started successfully');
            } else {
                const error = await response.json();
                this.showError(error.error || 'Failed to start simulation');
            }
        } catch (error) {
            console.error('Failed to start simulation:', error);
            this.showError('Failed to start simulation');
        }
    }
    
    async stopSimulation() {
        try {
            const response = await fetch('/api/stop', {
                method: 'POST'
            });
            
            if (response.ok) {
                this.isRunning = false;
                this.updateUIForStoppedState();
                this.stopStatusUpdates();
                this.showSuccess('Simulation stopped');
            } else {
                this.showError('Failed to stop simulation');
            }
        } catch (error) {
            console.error('Failed to stop simulation:', error);
            this.showError('Failed to stop simulation');
        }
    }
    
    updateUIForRunningState() {
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
        document.getElementById('export-btn').disabled = true;
        
        // Show simulation status panel
        const statusPanel = document.getElementById('simulation-status');
        statusPanel.style.display = 'block';
        statusPanel.classList.add('fade-in');
        
        // Hide results panel
        document.getElementById('results-panel').style.display = 'none';
    }
    
    updateUIForStoppedState() {
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        document.getElementById('export-btn').disabled = false;
        
        // Hide FL status panel
        document.getElementById('fl-status').style.display = 'none';
    }
    
    startStatusUpdates() {
        this.statusInterval = setInterval(() => {
            this.updateStatus();
        }, 1000);
    }
    
    stopStatusUpdates() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            if (!status.is_running && this.isRunning) {
                // Simulation finished
                this.isRunning = false;
                this.updateUIForStoppedState();
                this.stopStatusUpdates();
                this.loadResults();
                this.showSuccess('Simulation completed successfully');
            }
            
            if (status.current_results) {
                this.updateSimulationDisplay(status.current_results);
            }
            
        } catch (error) {
            console.error('Failed to update status:', error);
        }
    }
    
    updateSimulationDisplay(results) {
        // Update phase indicator
        const phaseText = results.phase === 1 ? 'Phase 1: Baseline' : 'Phase 2: Federated Learning';
        document.getElementById('current-phase').textContent = phaseText;
        
        // Update progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const progress = Math.round(results.progress);
        
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;
        
        // Update metrics
        if (results.metrics) {
            this.updateMetricDisplay('metric-latency', results.metrics.average_latency || 0, 'ms');
            this.updateMetricDisplay('metric-loss', results.metrics.packet_loss_rate || 0, '%', 100);
            this.updateMetricDisplay('metric-cache', results.metrics.cache_hit_rate || 0, '%', 100);
            this.updateMetricDisplay('metric-throughput', results.metrics.throughput || 0, 'Mbps');
        }
        
        // Show FL status for phase 2
        if (results.phase === 2) {
            this.showFLStatus(results);
        }
    }
    
    updateMetricDisplay(elementId, value, unit, multiplier = 1) {
        const element = document.getElementById(elementId);
        const displayValue = (value * multiplier).toFixed(2);
        element.textContent = `${displayValue} ${unit}`;
    }
    
    showFLStatus(results) {
        const flPanel = document.getElementById('fl-status');
        flPanel.style.display = 'block';
        flPanel.classList.add('fade-in');
        
        if (results.fl_round !== undefined) {
            document.getElementById('fl-round').textContent = results.fl_round;
        }
        
        if (results.training_history) {
            const history = results.training_history;
            document.getElementById('fl-accuracy').textContent = (history.accuracy * 100).toFixed(1) + '%';
            document.getElementById('fl-loss').textContent = history.loss.toFixed(3);
            document.getElementById('fl-participants').textContent = history.participants;
            
            this.updateTrainingChart(results.fl_round, history);
        }
    }
    
    updateTrainingChart(round, history) {
        if (!this.trainingChart) return;
        
        const data = this.trainingChart.data;
        
        // Add new data point
        data.labels.push(`Round ${round}`);
        data.datasets[0].data.push(history.accuracy);
        data.datasets[1].data.push(history.loss);
        
        // Keep only last 10 rounds for readability
        if (data.labels.length > 10) {
            data.labels.shift();
            data.datasets[0].data.shift();
            data.datasets[1].data.shift();
        }
        
        this.trainingChart.update();
    }
    
    async loadResults() {
        try {
            const response = await fetch('/api/results');
            const results = await response.json();
            
            this.displayResults(results);
        } catch (error) {
            console.error('Failed to load results:', error);
            this.showError('Failed to load results');
        }
    }
    
    displayResults(results) {
        const resultsPanel = document.getElementById('results-panel');
        resultsPanel.style.display = 'block';
        resultsPanel.classList.add('fade-in');
        
        // Hide simulation status
        document.getElementById('simulation-status').style.display = 'none';
        document.getElementById('fl-status').style.display = 'none';
        
        // Display comparative analysis
        if (results.comparative_analysis) {
            this.displayComparativeAnalysis(results.comparative_analysis);
        }
        
        // Display phase details
        this.displayPhaseDetails('phase1', results.phase1_metrics);
        this.displayPhaseDetails('phase2', results.phase2_metrics);
        
        // Display training history
        if (results.training_history) {
            this.displayTrainingHistory(results.training_history);
        }
    }
    
    displayComparativeAnalysis(analysis) {
        const container = document.getElementById('comparison-metrics');
        container.innerHTML = '';
        
        Object.keys(analysis).forEach(metric => {
            const data = analysis[metric];
            const metricDiv = document.createElement('div');
            metricDiv.className = 'comparison-metric';
            
            const improvementClass = data.improved ? 'positive' : 'negative';
            const improvementText = data.improvement_percent >= 0 ? '+' : '';
            
            metricDiv.innerHTML = `
                <div class="metric-name">${this.formatMetricName(metric)}</div>
                <div class="metric-comparison">
                    <div class="metric-values">
                        <span class="phase-value phase1-value">${data.phase1_mean.toFixed(3)}</span>
                        <span class="phase-value phase2-value">${data.phase2_mean.toFixed(3)}</span>
                    </div>
                    <div class="improvement ${improvementClass}">
                        ${improvementText}${data.improvement_percent.toFixed(1)}%
                    </div>
                </div>
            `;
            
            container.appendChild(metricDiv);
        });
    }
    
    displayPhaseDetails(phase, metrics) {
        const container = document.getElementById(`${phase}-details`);
        container.innerHTML = '';
        
        if (!metrics) {
            container.innerHTML = '<p>No data available for this phase.</p>';
            return;
        }
        
        Object.keys(metrics).forEach(metric => {
            const data = metrics[metric];
            if (typeof data === 'object' && data.mean !== undefined) {
                const metricDiv = document.createElement('div');
                metricDiv.className = 'metric-summary';
                metricDiv.innerHTML = `
                    <h5>${this.formatMetricName(metric)}</h5>
                    <p>Mean: ${data.mean.toFixed(3)}, Std: ${data.std.toFixed(3)}</p>
                    <p>Range: ${data.min.toFixed(3)} - ${data.max.toFixed(3)}</p>
                `;
                container.appendChild(metricDiv);
            }
        });
    }
    
    displayTrainingHistory(history) {
        const container = document.getElementById('training-details');
        container.innerHTML = '';
        
        // Create summary
        const summary = document.createElement('div');
        summary.className = 'training-summary';
        summary.innerHTML = `
            <h5>Training Summary</h5>
            <p>Total Rounds: ${history.length}</p>
            <p>Final Accuracy: ${(history[history.length - 1]?.accuracy * 100 || 0).toFixed(1)}%</p>
            <p>Final Loss: ${(history[history.length - 1]?.loss || 0).toFixed(3)}</p>
        `;
        container.appendChild(summary);
        
        // Create progress chart
        this.createProgressChart(history);
    }
    
    createProgressChart(history) {
        const canvas = document.getElementById('training-progress-chart');
        if (!canvas) return;
        
        new Chart(canvas, {
            type: 'line',
            data: {
                labels: history.map(h => `Round ${h.round}`),
                datasets: [{
                    label: 'Model Accuracy',
                    data: history.map(h => h.accuracy),
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Training Loss',
                    data: history.map(h => h.loss),
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Complete Training Progress'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Accuracy'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        min: 0,
                        max: 2,
                        title: {
                            display: true,
                            text: 'Loss'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }
    
    formatMetricName(metric) {
        return metric.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    
    async exportResults() {
        try {
            const response = await fetch('/api/export/json');
            const results = await response.json();
            
            const blob = new Blob([JSON.stringify(results, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ndn_fdrl_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showSuccess('Results exported successfully');
        } catch (error) {
            console.error('Failed to export results:', error);
            this.showError('Failed to export results');
        }
    }
    
    showSuccess(message) {
        this.showMessage(message, 'success');
    }
    
    showError(message) {
        this.showMessage(message, 'error');
    }
    
    showMessage(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            background: ${type === 'success' ? '#27ae60' : '#e74c3c'};
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 5000);
    }
}

// Global functions for tab management
function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(`${tabName}-tab`).style.display = 'block';
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Initialize the simulation interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    new SimulationInterface();
});
