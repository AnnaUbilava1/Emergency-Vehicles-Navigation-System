import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import requests

import time
from datetime import datetime
import warnings

# Optional interactive map visualization
try:
    import folium
    from folium import plugins
except ImportError:
    folium = None

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from visualisations import *
warnings.filterwarnings('ignore')

# System configuration
CONFIG = {
    'google_maps_api_key': 'AIzaSyDHJVrDbDje8-R6HsuuaOQhnM5L9ZggZhI', # i know this shouldn't have been pushed, but after the presentation we changed the key 
    'use_mock_data': False,  # Set to False for using real API keys
    'emergency_speed_bonus': 0.7,  #  1.42x faster 
    'traffic_light_priority': 0.8,  # 20% time reduction due to traffic light priority
    'emergency_lane_access': 0.7,  # 30% time reduction due to emergency lane access
}


class RealTrafficAPI:
    """
    Real traffic API integration with Google Maps and HERE Maps
    """

    def __init__(self, google_key=None, here_key=None):
        self.google_key = google_key
        self.here_key = here_key
        self.cache = {}  # Cache API responses to avoid rate limiting

    def get_google_traffic_data(self, origin, destination):
        """Get traffic data from Google Maps API"""
        if not self.google_key or self.google_key == 'AIzaSyDHJVrDbDje8-R6HsuuaOQhnM5L9ZggZhI':
            return None

        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': f"{origin[0]},{origin[1]}",
            'destination': f"{destination[0]},{destination[1]}",
            'departure_time': 'now',
            'traffic_model': 'best_guess',
            'key': self.google_key
        }

        try:
            response = requests.get(base_url, params=params)
            data = response.json()

            if data['status'] == 'OK':
                route = data['routes'][0]
                duration = route['legs'][0]['duration']['value']
                duration_in_traffic = route['legs'][0].get('duration_in_traffic', {}).get('value', duration)

                # Calculate traffic impact multiplier
                traffic_multiplier = duration_in_traffic / duration if duration > 0 else 1.0
                return traffic_multiplier

        except Exception as e:
            print(f"Google API error: {e}")

        return None

    def get_traffic_multiplier(self, edge):
        """Get traffic multiplier with fallback to mock data"""
        cache_key = f"{edge[0]}_{edge[1]}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        multiplier = None
        
        if not CONFIG['use_mock_data']:
            # Convert grid coordinates to approximate lat/lng
            origin_lat = 40.7128 + edge[0][0] * 0.01  # Approximate NYC coordinates
            origin_lng = -74.0060 + edge[0][1] * 0.01
            dest_lat = 40.7128 + edge[1][0] * 0.01
            dest_lng = -74.0060 + edge[1][1] * 0.01

            # Try Google first, then HERE
            multiplier = self.get_google_traffic_data(
                (origin_lat, origin_lng), (dest_lat, dest_lng)
            )
            
        # Fallback to mock data if API fails or not configured
        if CONFIG['use_mock_data'] or multiplier is None:
            multiplier = self._generate_realistic_traffic_data(edge)

        # Cache the result
        self.cache[cache_key] = multiplier
        return multiplier

    def _generate_realistic_traffic_data(self, edge):
        """Generate realistic traffic data based on time and location patterns"""
        current_hour = datetime.now().hour

        # Time-based traffic patterns
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19: #Rush hours
            base_multiplier = np.random.uniform(1.8, 3.5)
        elif 10 <= current_hour <= 16:  # Daytime
            base_multiplier = np.random.uniform(1.2, 2.0)
        else:  # Night/early morning
            base_multiplier = np.random.uniform(1.0, 1.3)

        # Location-based factors (center vs edges) (simulate downtown vs suburbs)
        location_factor = 1.0
        is_center = any(abs(node[0] - 3.5) < 1.5 and abs(node[1] - 3.5) < 1.5 for node in edge)
        if is_center:
            location_factor = 1.4

        # Random event factor (accidents, construction, etc.)
        event_factor = np.random.choice([1.0, 2.5], p=[0.99, 0.01])

        return base_multiplier * location_factor * event_factor


class BayesianPredictor:
    """
    Bayesian Network for traffic prediction based on multiple factors
    """

    def __init__(self):
        # Define network structure
        self.model = DiscreteBayesianNetwork([
            ('TimeOfDay', 'Congestion'),
            ('Weather', 'Congestion'),
            ('Accident', 'Congestion'),
            ('RoadType', 'Congestion'),
            ('Emergency', 'ResponseTime'),
            ('Congestion', 'ResponseTime')
        ])

        # Define probability distributions
        cpd_time = TabularCPD(
            variable='TimeOfDay', variable_card=3,
            values=[[0.25], [0.5], [0.25]],
            state_names={'TimeOfDay': ['Rush', 'Normal', 'Night']}
        )

        cpd_weather = TabularCPD(
            variable='Weather', variable_card=2,
            values=[[0.3], [0.7]],
            state_names={'Weather': ['Bad', 'Good']}
        )

        cpd_accident = TabularCPD(
            variable='Accident', variable_card=2,
            values=[[0.05], [0.95]],
            state_names={'Accident': ['Yes', 'No']}
        )

        cpd_roadtype = TabularCPD(
            variable='RoadType', variable_card=2,
            values=[[0.4], [0.6]],
            state_names={'RoadType': ['Highway', 'Local']}
        )

        # Complex congestion model considering all factors
        cpd_congestion = TabularCPD(
            variable='Congestion', variable_card=3,
            values=[
                # High congestion probabilities
                [0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.3, 0.2, 0.1, 0.05,
                 0.4, 0.3, 0.2, 0.1],
                # Medium congestion probabilities
                [0.15, 0.2, 0.25, 0.3, 0.08, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.3, 0.35, 0.4, 0.45, 0.4, 0.45,
                 0.5, 0.55, 0.3, 0.35, 0.4, 0.45],
                # Low congestion probabilities
                [0.05, 0.1, 0.15, 0.2, 0.02, 0.05, 0.1, 0.15, 0.3, 0.35, 0.4, 0.45, 0.2, 0.25, 0.3, 0.35, 0.3, 0.35,
                 0.4, 0.4, 0.3, 0.35, 0.4, 0.45]
            ],
            evidence=['TimeOfDay', 'Weather', 'Accident', 'RoadType'],
            evidence_card=[3, 2, 2, 2],
            state_names={
                'Congestion': ['High', 'Medium', 'Low'],
                'TimeOfDay': ['Rush', 'Normal', 'Night'],
                'Weather': ['Bad', 'Good'],
                'Accident': ['Yes', 'No'],
                'RoadType': ['Highway', 'Local']
            }
        )

        cpd_emergency = TabularCPD(
            variable='Emergency', variable_card=2,
            values=[[0.2], [0.8]],
            state_names={'Emergency': ['Yes', 'No']}
        )

        # Response time model based on emergency status and congestion
        cpd_response = TabularCPD(
            variable='ResponseTime', variable_card=3,
            values=[
                # P(Fast, Normal, Slow | Emergency, Congestion)
                # Fast, Normal, Slow response times
                [0.8, 0.6, 0.4, 0.05, 0.1, 0.2],
                [0.15, 0.3, 0.4, 0.25, 0.3, 0.4],
                [0.05, 0.1, 0.2, 0.7, 0.6, 0.4]
            ],
            evidence=['Emergency', 'Congestion'],
            evidence_card=[2, 3],
            state_names={
                'ResponseTime': ['Fast', 'Normal', 'Slow'],
                'Emergency': ['Yes', 'No'],
                'Congestion': ['High', 'Medium', 'Low']
            }
        )

        self.model.add_cpds(cpd_time, cpd_weather, cpd_accident, cpd_roadtype,
                            cpd_congestion, cpd_emergency, cpd_response)

        assert self.model.check_model()
        self.inference = VariableElimination(self.model)
        print("Enhanced Bayesian Network initialized successfully.")

    def predict_congestion_score(self, evidence):
        """Return a congestion score between 0 and 1"""
        query = self.inference.query(variables=['Congestion'], evidence=evidence)

        high_prob = query.values[query.state_names['Congestion'].index('High')]
        medium_prob = query.values[query.state_names['Congestion'].index('Medium')]

        # Weighted score where 1.0 = maximum congestion, 0 = no congestion
        score = (high_prob * 1.0 + medium_prob * 0.5)
        return score


class SmartEmergencyNavigation:
    """
    Enhanced emergency navigation system with real-time optimization
    """

    def __init__(self, graph, traffic_api, bayesian_predictor):
        self.graph = graph.copy()
        self.traffic_api = traffic_api
        self.bayesian_predictor = bayesian_predictor
        self.route_history = []

    def _calculate_dynamic_weight(self, u, v, data, evidence, is_emergency=True):
        """Calculate optimized weight for emergency vs regular vehicles"""
        base_weight = data['base_weight']
        
        # Get real-time traffic multiplier
        traffic_multiplier = self.traffic_api.get_traffic_multiplier((u, v))
    
        # Get Bayesian congestion prediction
        road_type = 'Highway' if data.get('road_type') == 'highway' else 'Local'
        full_evidence = {**evidence, 'RoadType': road_type}
        full_evidence['Emergency'] = 'Yes' if is_emergency else 'No'
        
        congestion_score = self.bayesian_predictor.predict_congestion_score(full_evidence)
        
        # Add accident-specific penalties to create route differentiation
        accident_penalty = 0
        if evidence.get('Accident') == 'Yes':
            # Spatially varying accident impact; heavy/medium penalty for center/near center routes
            center_penalty = 0
            for node in [u, v]:
                if 2 <= node[0] <= 5 and 2 <= node[1] <= 5:
                    center_penalty += 8.0
                elif 1 <= node[0] <= 6 and 1 <= node[1] <= 6:
                    center_penalty += 4.0
            
            accident_penalty = center_penalty
    
        # Calculate final weight based on vehicle type
        if is_emergency:
            # Emergency vehicles get significant advantages
            emergency_factor = (
                CONFIG['emergency_speed_bonus'] * 0.5 *
                CONFIG['traffic_light_priority'] * 0.6 *
                CONFIG['emergency_lane_access'] * 0.5
            )
        
            # Emergency vehicles can bypass some congestion and accidents
            effective_traffic = 1 + (traffic_multiplier - 1) * emergency_factor
            effective_congestion = congestion_score * emergency_factor * 0.3
            effective_accident = accident_penalty * 0.2
        
            dynamic_weight = (base_weight * max(0.3, effective_traffic) + 
                             effective_congestion * 3 + effective_accident)
        else:
            # Regular vehicles are more affected by all conditions
            congestion_multiplier = 1 + congestion_score * 2.0
            accident_multiplier = 1 + (accident_penalty / base_weight) * 0.8
            
            dynamic_weight = (base_weight * traffic_multiplier * congestion_multiplier * 
                             accident_multiplier + accident_penalty)
    
        # Store analysis data
        self.graph[u][v]['dynamic_weight'] = dynamic_weight
        self.graph[u][v]['traffic_multiplier'] = traffic_multiplier
        self.graph[u][v]['congestion_score'] = congestion_score
    
        return max(0.1, dynamic_weight)

    def find_emergency_route(self, start, end, evidence, algorithm='a_star'):
        """Find optimal emergency route using specified algorithm"""
        print(f"\n🚨 EMERGENCY ROUTE CALCULATION 🚨")
        print(f"From: {start} → To: {end}")
        print(f"Conditions: {evidence}")

        # Emergency vehicle weight function
        weight_func = lambda u, v, d: self._calculate_dynamic_weight(u, v, d, evidence, is_emergency=True)

        # Heuristic for A* (Euclidean distance)
        heuristic = lambda a, b: math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        start_time = time.time()

        if algorithm == 'a_star':
            path = nx.astar_path(self.graph, start, end, heuristic=heuristic, weight=weight_func)
            total_time = nx.astar_path_length(self.graph, start, end, heuristic=heuristic, weight=weight_func)
        else:
            path = nx.dijkstra_path(self.graph, start, end, weight=weight_func)
            total_time = nx.dijkstra_path_length(self.graph, start, end, weight=weight_func)

        calculation_time = time.time() - start_time

        # Store route for analysis
        route_info = {
            'path': path,
            'total_time': total_time,
            'calculation_time': calculation_time,
            'algorithm': algorithm,
            'evidence': evidence,
            'timestamp': datetime.now()
        }
        self.route_history.append(route_info)

        return path, total_time

    def find_regular_route(self, start, end, evidence):
        """Find regular navigation route for comparison"""
        weight_func = lambda u, v, d: self._calculate_dynamic_weight(u, v, d, evidence, is_emergency=False)
        path = nx.dijkstra_path(self.graph, start, end, weight=weight_func)
        total_time = nx.dijkstra_path_length(self.graph, start, end, weight=weight_func)
        return path, total_time
    
    def find_alternative_route(self, start, end, evidence, avoid_center=True):
        """Find alternative route that avoids center areas"""
        print(f"\n🔄 FINDING ALTERNATIVE ROUTE...")

        # Temporarily modify weights to discourage center routes
        temp_weights = {}
        if avoid_center:
           for u, v, data in self.graph.edges(data=True):
                temp_weights[(u, v)] = data.get('dynamic_weight', data['base_weight'])
            
                # Add penalty for center routes
                center_penalty = 0
                for node in [u, v]:
                   if 2 <= node[0] <= 5 and 2 <= node[1] <= 5:
                       center_penalty += 15.0
            
                data['temp_weight'] = temp_weights[(u, v)] + center_penalty
       
        # Find route using temporary weights
        weight_func = lambda u, v, d: d.get('temp_weight', d.get('dynamic_weight', d['base_weight']))
        try:
            path = nx.dijkstra_path(self.graph, start, end, weight=weight_func)
            total_time = nx.dijkstra_path_length(self.graph, start, end, weight=weight_func)
        except nx.NetworkXNoPath:
           # Fallback to regular route if no alternative found
           return self.find_regular_route(start, end, evidence)
    
        # Clean up temporary weights
        for u, v, data in self.graph.edges(data=True):
            if 'temp_weight' in data:
                del data['temp_weight']
    
        return path, total_time


def create_realistic_city_graph():
    """Create a realistic city road network with varied connections"""
    G = nx.Graph()
    
    # Create 8x8 grid for complexity
    nodes = [(x, y) for x in range(8) for y in range(8)]
    G.add_nodes_from(nodes)
    
    # Add regular streets with location-based weight variation
    for x in range(8):
        for y in range(8):
            if x < 7:
                # Center area is slower (more congested)
                if 2 <= x <= 4 and 2 <= y <= 4:
                    weight = np.random.uniform(2.5, 4.5)
                else:
                    weight = np.random.uniform(1.5, 3.0)
                G.add_edge((x, y), (x + 1, y), base_weight=weight, distance=1.0, road_type='local')
            if y < 7:
                if 2 <= x <= 4 and 2 <= y <= 4:
                    weight = np.random.uniform(2.5, 4.5)
                else:
                    weight = np.random.uniform(1.5, 3.0)
                G.add_edge((x, y), (x, y + 1), base_weight=weight, distance=1.0, road_type='local')
    
    # Add highways on edges for alternative fast routes
    for i in [1, 6]:  # Horizontal highways on rows 1 and 6 (edges)
        for j in range(7):
            G.add_edge((j, i), (j + 1, i), base_weight=0.6, distance=1.2, road_type='highway')
    
    for i in [1, 6]:  # Vertical highways on columns 1 and 6 (edges)
        for j in range(7):
            G.add_edge((i, j), (i, j + 1), base_weight=0.6, distance=1.2, road_type='highway')
    
    # Add some diagonal connections (bridges/overpasses) for more path options
    for x in range(0, 7, 2):
        for y in range(0, 7, 2):
            if x + 1 < 8 and y + 1 < 8:
                G.add_edge((x, y), (x + 1, y + 1), base_weight=2.0, distance=1.4, road_type='bridge')
            if x + 1 < 8 and y - 1 >= 0:
                G.add_edge((x, y), (x + 1, y - 1), base_weight=2.0, distance=1.4, road_type='bridge')
    
    print(f"Enhanced city graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def run_comprehensive_simulation():
    """Run complete simulation with route analysis"""
    print("🚨 INITIALIZING SMART EMERGENCY NAVIGATION SYSTEM 🚨\n")
    
    city_graph = create_realistic_city_graph()
    traffic_api = RealTrafficAPI()
    bayesian_predictor = BayesianPredictor()
    emergency_nav = SmartEmergencyNavigation(city_graph, traffic_api, bayesian_predictor)
    
    start_node = (0, 0)
    end_node = (7, 7)
    
    # Determine current traffic conditions
    current_hour = datetime.now().hour
    time_condition = 'Rush' if 7 <= current_hour <= 9 or 17 <= current_hour <= 19 else 'Normal'
    initial_evidence = {'TimeOfDay': time_condition, 'Weather': 'Good', 'Accident': 'No'}
    
    print(f"📍 Emergency Scenario: From {start_node} to {end_node} during '{time_condition}' hours.")
    
    # Calculate initial routes
    emergency_path, emergency_time = emergency_nav.find_emergency_route(start_node, end_node, initial_evidence)
    regular_path, regular_time = emergency_nav.find_regular_route(start_node, end_node, initial_evidence)
    
    analysis_graph = emergency_nav.graph
    
    routes_data = {
        'Emergency Route': {'path': emergency_path, 'time': emergency_time, 'color': '#2ca02c', 'style': 'solid', 'width': 4},
        'Regular Route': {'path': regular_path, 'time': regular_time, 'color': '#d62728', 'style': 'dashed', 'width': 3}
    }
    
    # Print initial results
    initial_time_saved = regular_time - emergency_time
    initial_percentage_improvement = (initial_time_saved / regular_time) * 100 if regular_time > 0 else 0
    print(f"\n📊 INITIAL ROUTE PERFORMANCE (Before Accident)")
    print(f"   - Regular Route Time:         {regular_time:.2f} minutes")
    print(f"   - Emergency Route Time:       {emergency_time:.2f} minutes")
    print(f"   - Initial Time Saved:         {initial_time_saved:.2f} minutes ({initial_percentage_improvement:.1f}% improvement)")
    
    # Simulate accident scenario
    print(f"\n🚨 SIMULATING MID-ROUTE ACCIDENT...")
    accident_evidence = {**initial_evidence, 'Accident': 'Yes'}
    
    # Calculate different route types under accident conditions
    # Emergency reroute (with accident conditions)
    reroute_path, reroute_time = emergency_nav.find_emergency_route(start_node, end_node, accident_evidence)
    
    # Regular route under accident (will avoid center due to penalties)
    regular_accident_path, regular_accident_time = emergency_nav.find_regular_route(start_node, end_node, accident_evidence)
    
    # Alternative route (explicitly avoids center for demonstration)
    alt_path, alt_time = emergency_nav.find_alternative_route(start_node, end_node, accident_evidence, avoid_center=True)
    
    analysis_graph = emergency_nav.graph
    
    # Update routes data with accident scenario results
    routes_data['Rerouted Emergency'] = {'path': reroute_path, 'time': reroute_time, 'color': '#1f77b4', 'style': 'dotted', 'width': 3.5}
    routes_data['Regular (Post-Accident)'] = {'path': regular_accident_path, 'time': regular_accident_time, 'color': '#ff7f0e', 'style': 'dashdot', 'width': 2.5}
    
    # Add alternative route if it's different from others
    if alt_path != reroute_path and alt_path != regular_accident_path:
        routes_data['Alternative Route'] = {'path': alt_path, 'time': alt_time, 'color': '#9467bd', 'style': 'solid', 'width': 2}
    
    final_emergency_time = reroute_time
    final_regular_time = regular_accident_time
    
    print(f"   - Regular route time increased from {regular_time:.2f} to {regular_accident_time:.2f} minutes")
    print(f"   - Emergency route time changed from {emergency_time:.2f} to {reroute_time:.2f} minutes")
    if 'Alternative Route' in routes_data:
        print(f"   - Alternative route time: {alt_time:.2f} minutes")
    
    # Final performance analysis
    print(f"\n{'=' * 60}")
    print("📊 FINAL PERFORMANCE ANALYSIS")
    time_saved = final_regular_time - final_emergency_time
    percentage_improvement = (time_saved / final_regular_time) * 100 if final_regular_time > 0 else 0
    
    print(f"   - Initial Regular Time:       {regular_time:.2f} minutes")
    print(f"   - Initial Emergency Time:     {emergency_time:.2f} minutes")
    print(f"   - Final Regular Time:         {final_regular_time:.2f} minutes (with accident)")
    print(f"   - Final Emergency Time:       {final_emergency_time:.2f} minutes (rerouted)")
    print(f"   - Time Saved:                 {time_saved:.2f} minutes ({percentage_improvement:.1f}% improvement)")
    
    # Generate all visualization plots
    plot_initial_route_comparison(routes_data)
    plot_route_comparison_map(analysis_graph, routes_data, start_node, end_node)
    plot_traffic_heatmap(analysis_graph)
    plot_congestion_heatmap(analysis_graph)
    plot_response_time_comparison(routes_data)
    plot_performance_improvement(routes_data)
    plot_statistics_report(routes_data, start_node, end_node, initial_evidence)
    
    # Generate interactive map if folium is available
    if folium:
        create_interactive_map_visualization(routes_data, start_node, end_node)
    else:
        print("\nNote: 'folium' is not installed. Skipping interactive map. Run: pip install folium")


def create_interactive_map_visualization(routes_data, start, end):
    """Create an interactive Folium map visualization"""

    # Convert grid coordinates to lat/lng (approximate NYC area)
    def grid_to_latlng(coord):
        lat = 40.7128 + coord[0] * 0.005
        lng = -74.0060 + coord[1] * 0.005
        return [lat, lng]

    # Create base map
    center_lat = 40.7128 + 3.5 * 0.005
    center_lng = -74.0060 + 3.5 * 0.005
    m = folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles='CartoDB positron')

    # Add route polylines
    for route_name, route_data in routes_data.items():
        path_coords = [grid_to_latlng(coord) for coord in route_data['path']]
        folium.PolyLine(
            locations=path_coords,
            color=route_data['color'],
            weight=5,
            opacity=0.8,
            popup=f"<b>{route_name}</b><br>Time: {route_data['time']:.1f} min"
        ).add_to(m)

    # Add start and end markers
    folium.Marker(
        location=grid_to_latlng(start),
        popup="<b>Emergency Start</b>",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)

    folium.Marker(
        location=grid_to_latlng(end),
        popup="<b>Emergency Destination</b>",
        icon=folium.Icon(color='red', icon='hospital', prefix='fa')
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Save interactive map
    map_filename = 'emergency_navigation_map.html'
    m.save(map_filename)
    print(f"\n🗺️  Interactive map saved as '{map_filename}'")


def setup_real_apis():
    """Display instructions for setting up real traffic APIs"""
    print(f"\n🔧 SETTING UP REAL TRAFFIC APIs")
    print(f"{'=' * 60}")
    print(f"""
To use real traffic data instead of mock data:

1. 🗝️  GET API KEYS:

   Google Maps API:
   • Go to: https://console.cloud.google.com/
   • Enable: Maps JavaScript API, Directions API
   • Create credentials and get your API key

   HERE Maps API (Alternative):  
   • Go to: https://developer.here.com/
   • Sign up for free developer account
   • Get your API key

2. 🔧 UPDATE CONFIGURATION:

   In the script, find the CONFIG dictionary and update it:
   CONFIG = {{
       'google_maps_api_key': 'YOUR_ACTUAL_GOOGLE_KEY_HERE',
       'here_api_key': 'YOUR_ACTUAL_HERE_KEY_HERE', 
       'use_mock_data': False,
       ...
   }}

3. 📦 INSTALL REQUIRED PACKAGES:

   pip install requests

4. 🚀 BENEFITS OF REAL APIs:
   • Live traffic conditions
   • Actual road closures and incidents  
   • Real construction delays
   • Weather-based traffic impacts

⚠️  Note: API calls may have usage limits and costs.
For development/testing, mock data provides a realistic simulation.
""")


if __name__ == '__main__':
    print("🚨 SMART EMERGENCY NAVIGATION SYSTEM v2.0 🚨")
    print("=" * 60)

    try:
        run_comprehensive_simulation()

        if CONFIG['use_mock_data']:
            setup_real_apis()

        print(f"\n✅ SIMULATION COMPLETED SUCCESSFULLY!")
        print(f"📊 Check the generated visualizations and reports above.")

    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()