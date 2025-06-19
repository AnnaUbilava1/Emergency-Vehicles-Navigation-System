# Smart Emergency Navigation System

## Overview

The Smart Emergency Navigation System aims to provide real-time optimized routing for emergency vehicles, considering traffic conditions, road types, weather, accidents, and other factors that influence travel time. The system uses a combination of Bayesian networks and traffic data from APIs (Google Maps or mock data) to calculate the best routes for emergency vehicles, minimizing response times during critical situations.

## Features

- **Real-time traffic prediction** using Google Maps and mock data.
- **Bayesian network** for predicting congestion levels based on factors like time of day, weather, and accidents.
- **Optimized route calculation** using A\* or Dijkstra's algorithm for emergency vehicles.
- **Interactive visualization** of routes and congestion on a map.
- **Performance comparison** between emergency routes and regular traffic routes.

## Installation

To run the system, you'll need to install several dependencies. Follow the steps below to set up your environment.

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/smart-emergency-navigation.git
cd smart-emergency-navigation
```

### 2. Create a virtual environment (optional but recommended)

```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

If you want to use real-time traffic data from Google Maps or HERE Maps, you will need to set up API keys.

- **Google Maps API**:

  Visit the Google Cloud Console, enable the Maps JavaScript API and Directions API, and create an API key.

- **HERE Maps API (optional)** :

  Visit the HERE Developer Portal, sign up for a free account, and get your API key.

### 5. Configure your API keys

Open the new.py file and replace the placeholder Google Maps API key in the CONFIG dictionary with your actual API key:

```py
CONFIG = {
    'google_maps_api_key': 'YOUR_GOOGLE_API_KEY',  # Replace with your actual API key
    'use_mock_data': False,  # Set to False for real traffic data
}
```

## Usage

### 1. Run the system:

To start the simulation and see the optimized emergency routes, run the following command:

```bash
python new.py
```

This will calculate and print the emergency route, regular route, and performance comparison, as well as generate visualizations for the routes and congestion.

### 2. Interactive map:

If you have folium installed and the use_mock_data flag is set to False, an interactive map showing the emergency routes will be generated. The map will be saved as emergency_navigation_map.html.

## Features Explained

- **Bayesian Network**:
  The system uses a Bayesian network to predict congestion based on several factors like time of day, weather, accidents, and road type. The network’s output helps adjust the route calculation for emergency vehicles.

- **Traffic Data API**:
  Traffic data is fetched in real-time from Google Maps or HERE Maps APIs. If no real API key is provided, mock data is used for traffic prediction.

- **Route Calculation**:
  The system uses A\* algorithm or Dijkstra’s algorithm to find the shortest path, considering dynamic weights for traffic and congestion.

## Project Structure

- **new.py** : The main script that runs the simulation, calculates routes, and generates visualizations.

- **requirements.txt** : A file containing all the dependencies required to run the system.

- **visualisations.py**: A module responsible for generating plots and maps.


## Visualization

The project includes several functions for visualizing the results of the emergency navigation system's calculations. These visualizations help to understand the traffic conditions, route comparisons, and overall system performance.

### Visualization Functions in `visualisations.py`

1. **Route Comparison Map (`plot_route_comparison_map`)**
   - This function creates a detailed map comparing the optimal routes for emergency vehicles and regular vehicles.
   - It displays the road network with edge weights representing the base travel time and highlights the selected routes with different styles and colors.
   - **Parameters**:
     - `graph`: The road network graph.
     - `routes_data`: Data for the different routes, including paths, colors, and times.
     - `start`, `end`: The start and end nodes for the route.
     - `filename`: The name of the file to save the visualization as.
   - **Output**: A map with route comparison, edge weights, and labeled start/end points.

2. **Traffic Heatmap (`plot_traffic_heatmap`)**
   - This heatmap visualizes the traffic intensity on the city grid. It uses traffic multipliers to represent congestion levels on each segment of the road network.
   - **Parameters**:
     - `graph`: The road network graph.
     - `filename`: The name of the file to save the heatmap.
   - **Output**: A heatmap where redder areas represent higher traffic and congestion.

3. **Congestion Heatmap (`plot_congestion_heatmap`)**
   - This heatmap shows the predicted congestion levels based on the Bayesian network's analysis. It uses congestion scores to visualize potential delays.
   - **Parameters**:
     - `graph`: The road network graph.
     - `filename`: The name of the file to save the heatmap.
   - **Output**: A heatmap where darker blue indicates higher congestion risk.

4. **Response Time Comparison (`plot_response_time_comparison`)**
   - This function compares the calculated travel times for different routes (emergency vs regular).
   - **Parameters**:
     - `routes_data`: Data for the different routes, including times and colors.
     - `filename`: The name of the file to save the chart.
   - **Output**: A bar chart showing the response time comparison.

5. **Performance Improvement (`plot_performance_improvement`)**
   - This function visualizes the performance improvement of the AI system compared to regular navigation, based on the time saved by the emergency system.
   - **Parameters**:
     - `routes_data`: Data for the routes, including times and colors.
     - `filename`: The name of the file to save the chart.
   - **Output**: A bar chart comparing regular and AI-optimized navigation, showing the time saved.

6. **Statistics Report (`plot_statistics_report`)**
   - This function generates a textual summary of the simulation's results, including detailed statistics for the routes and performance improvements.
   - **Parameters**:
     - `routes_data`: Data for the routes, including times.
     - `start`, `end`: The start and end nodes of the route.
     - `evidence`: The evidence (conditions) used in the simulation.
     - `filename`: The name of the file to save the report.
   - **Output**: A textual report summarizing the simulation results.

### How to Use the Visualization Functions

To generate the visualizations, simply call the corresponding function in `visualisations.py`. Each function takes data from the simulation and generates a plot or heatmap, which is saved as a PNG file. For example:

```python
from visualisations import plot_route_comparison_map

# Assuming `graph`, `routes_data`, `start`, and `end` are defined
plot_route_comparison_map(graph, routes_data, start, end, "route_comparison.png")
```
This will create a visualization of the route comparison and save it as route_comparison.png.

### Visualization Output Files
The visualizations are saved as PNG files by default, but you can customize the filenames for each plot as needed. You can view the resulting images to analyze:

- Route comparisons between emergency and regular vehicles.

- Traffic heatmaps showing real-time congestion.

- AI performance improvements over standard navigation systems.


## Acknowledgements

**pgmpy**: A library for probabilistic graphical models used in this project for Bayesian networks.

**Google Maps API**: Used for real-time traffic data.

**Folium**: For creating interactive maps.

**NetworkX**: Used for graph operations.
