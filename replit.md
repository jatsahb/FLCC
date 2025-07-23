# NDN-FDRL Simulation System

## Overview

This repository contains a comprehensive simulation system for Named Data Networking (NDN) with Federated Deep Reinforcement Learning (FDRL) for congestion control. The system simulates realistic network environments where multiple domains collaborate through federated learning to optimize network performance while preserving privacy.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The system follows a modular, layered architecture with clear separation of concerns:

### Frontend Layer
- **Web Interface**: Flask-based web application with HTML/CSS/JavaScript frontend
- **Alternative Interface**: Streamlit application for advanced visualization and analysis
- **Real-time Updates**: JavaScript-based real-time status monitoring and chart updates

### Core Simulation Layer
- **NDN Network Simulator**: Comprehensive NDN network simulation with realistic packet handling
- **DDPG Agent**: Deep Deterministic Policy Gradient reinforcement learning agent for congestion control
- **Federated Coordinator**: Manages federated learning across multiple network domains
- **Scenario Generator**: Creates realistic network congestion and failure scenarios
- **Stakeholder Monitor**: Models behavior of consumers, producers, and routers

### Utility Layer
- **Configuration Management**: Centralized configuration with validation
- **Metrics Collection**: Comprehensive metrics gathering and analysis
- **Privacy Protection**: Differential privacy implementation for federated learning
- **Visualization**: Network topology and performance visualization tools

## Key Components

### NDN Network Simulation
- **Packet Types**: Interest, Data, and NACK packet handling
- **Core Structures**: Pending Interest Table (PIT), Forwarding Information Base (FIB), Content Store (CS)
- **Network Entities**: Routers, consumers, producers with realistic behavior models
- **Traffic Patterns**: Configurable traffic loads and content popularity distributions

### Deep Reinforcement Learning
- **Algorithm**: DDPG (Deep Deterministic Policy Gradient) for continuous action spaces
- **State Space**: Network metrics including latency, congestion, cache hit rates
- **Action Space**: Congestion control parameters (window sizes, rate limits)
- **Experience Replay**: Buffer for storing and sampling training experiences
- **Noise Generation**: Ornstein-Uhlenbeck noise for exploration

### Federated Learning
- **Multi-Domain Architecture**: Support for 2-10 network domains
- **Model Aggregation**: FedAvg algorithm with privacy-preserving mechanisms
- **Privacy Protection**: Differential privacy with configurable epsilon/delta parameters
- **Participation Management**: Dynamic participant selection and contribution weighting

### Scenario Management
- **Predefined Scenarios**: Campus, IoT, Enterprise, CDN, and Emergency network scenarios
- **Dynamic Events**: Flash crowds, link failures, DDoS attacks, equipment failures
- **Severity Levels**: Low, medium, high, and critical congestion scenarios
- **Temporal Patterns**: Flash crowd, sustained, intermittent, and gradual patterns

## Data Flow

### Simulation Execution Flow
1. **Initialization**: Load configuration, initialize network topology and agents
2. **Phase 1**: Baseline simulation without federated learning
3. **Phase 2**: Enhanced simulation with federated learning enabled
4. **Metrics Collection**: Continuous monitoring and data collection
5. **Analysis**: Performance comparison and statistical analysis

### Federated Learning Flow
1. **Local Training**: Each domain trains DDPG agent on local network data
2. **Model Sharing**: Domains share model updates with privacy protection
3. **Aggregation**: Central coordinator aggregates models using FedAvg
4. **Distribution**: Updated global model distributed back to domains
5. **Iteration**: Process repeats for multiple federated rounds

### Data Processing Pipeline
1. **Real-time Collection**: Network metrics collected every 10 seconds
2. **Preprocessing**: Data normalization and feature engineering
3. **Storage**: Metrics stored in time series format
4. **Analysis**: Statistical analysis and performance comparison
5. **Visualization**: Real-time charts and performance dashboards

## External Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework for DDPG implementation
- **Flask**: Web application framework for user interface
- **NumPy/Pandas**: Numerical computing and data manipulation
- **Scikit-learn**: Machine learning utilities and metrics

### Visualization
- **Chart.js**: Frontend charting library for real-time visualization
- **Plotly**: Advanced plotting for Streamlit interface
- **Matplotlib**: Static plot generation for reports

### Optional Enhancements
- **Database Support**: The system is designed to work with databases (likely PostgreSQL with Drizzle ORM) for persistent storage, though currently uses in-memory storage
- **WebSocket Support**: For real-time communication between frontend and backend

## Deployment Strategy

### Development Environment
- **Local Execution**: Direct Python execution with Flask development server
- **Configuration**: YAML-based configuration files for easy parameter adjustment
- **Logging**: Comprehensive logging with configurable levels and file output

### Production Considerations
- **Scalability**: Modular design supports horizontal scaling of simulation components
- **Data Persistence**: Ready for database integration for long-term data storage
- **Privacy Compliance**: Built-in differential privacy for regulatory compliance
- **Monitoring**: Comprehensive metrics collection for performance monitoring

### Container Deployment
- The system is structured to support containerization with Docker
- Separate containers can be used for simulation engine, web interface, and data storage
- Environment variables and configuration files support different deployment environments

The architecture prioritizes modularity, extensibility, and realistic simulation of NDN networks with federated learning capabilities. The system can run standalone simulations or be extended for distributed deployment across multiple nodes.