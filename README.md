# 🚨 Smart Emergency Navigation System

## Overview

The Smart Emergency Navigation System aims to provide real-time optimized routing for emergency vehicles, considering traffic conditions, road types, weather, accidents, and other factors that influence travel time. The system uses a combination of Bayesian networks and traffic data from APIs (Google Maps or mock data) to calculate the best routes for emergency vehicles, minimizing response times during critical situations.

---

## 🚗 Problem Statement

Urban traffic congestion often causes critical delays for emergency vehicles like ambulances and fire trucks, putting lives and property at risk. Traditional GPS systems are reactive and do not prioritize emergency-specific needs such as signal preemption, lane privileges, or intelligent rerouting during incidents.

---

## Project Goals

- Design an AI system that **predicts**, **adapts**, and **reroutes** emergency vehicles in real time.
- Integrate real-world traffic APIs to dynamically reflect changing conditions.
- Use Bayesian reasoning to anticipate delays before they become visible.
- Simulate real-world response time improvements of 20–30%.

---

## AI & ML Techniques Used

### 🔍 Search Algorithms

- **A\*** and **Dijkstra** for shortest pathfinding on a city road network.
- **Dynamic edge weights** that reflect emergency speed, traffic lights, lane access, and congestion.

### Bayesian Network

- Predicts traffic congestion based on:
  - `TimeOfDay`, `Weather`, `Accident`, `RoadType`
- Provides a **congestion score** used to penalize risky routes.
- Enhances **reasoning under uncertainty** and supports proactive rerouting.

---

## System Components

| Component                        | Description                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| `SmartEmergencyNavigation`       | Core class that calculates optimal routes using AI logic            |
| `RealTrafficAPI`                 | Integrates with Google Maps or simulates realistic traffic patterns |
| `EnhancedBayesianPredictor`      | Predicts congestion probabilities dynamically                       |
| `visualisations.py`              | Generates heatmaps, charts, and route comparison plots              |
| `run_comprehensive_simulation()` | Orchestrates full simulation: before/after accident routing         |

---

## Demo: What It Does

- Calculates both **emergency** and **regular** routes from origin to destination.
- Injects a simulated **accident** mid-route and reroutes emergency vehicles.
- Generates visual comparisons:
  - Traffic heatmap
  - Bayesian congestion prediction
  - Route overlays
  - Time-saving bar charts

---

## 📈 Example Results

- 🕒 **42% faster** response time with emergency privileges and rerouting
- 🚧 Avoided central bottlenecks intelligently during accident simulation
- 📊 Visual evidence generated in plots: `plot_*.png`

---

## How to Run

### 1. Install Requirements

```bash
pip install networkx matplotlib numpy pgmpy requests folium
```

### 2. Run the Simulation

```bash
python new.py
```

This runs a full simulation and generates result plots in the working directory.

To use live traffic:

Get a Google Maps API key

In new.py, update:

```bash
CONFIG = {
    'google_maps_api_key': 'YOUR_KEY_HERE',
    'use_mock_data': False,
}
```

⚠️ API calls may be rate-limited. Mock traffic simulation is enabled by default.

### 👥 Team Contributions

| Member       | Contribution                                                                                       |
| ------------ | -------------------------------------------------------------------------------------------------- |
| **Member 1** | Implemented **A\*** and **Dijkstra**, created emergency routing logic                              |
| **Member 2** | Designed the **Bayesian Network** for predictive congestion modeling                               |
| **Member 3** | Built **Google Maps API** integration and fallback traffic simulator                               |
| **Member 4** | Created the **visualization and simulation framework**, including heatmaps and performance reports |

### 📁 Outputs

| File                            | Description                            |
| ------------------------------- | -------------------------------------- |
| `plot_0_initial_comparison.png` | Baseline vs Emergency route comparison |
| `plot_1_route_map.png`          | Route overlay on grid                  |
| `plot_2_traffic_heatmap.png`    | Real-time traffic heatmap              |
| `plot_3_congestion_heatmap.png` | AI-predicted congestion score heatmap  |
| `plot_4_time_comparison.png`    | Time bar charts for all routes         |
| `plot_5_performance.png`        | Performance gain after rerouting       |
| `plot_6_statistics.png`         | Summary report of times and conditions |

### Why This Matters

This system demonstrates how AI + real-world data can:

-Save lives with faster emergency response

-Handle uncertainty with intelligent planning

-Scale to city-wide public safety infrastructure
