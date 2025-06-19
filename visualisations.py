import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D  # Ensure this is imported for the legend handles


def save_plot(fig, filename):
    """Saves a plot to a file and prints a confirmation."""
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Visualization saved as '{filename}'")


def plot_route_comparison_map(graph, routes_data, start, end, filename="plot_1_route_map.png"):
    """Creates a detailed plot of the route comparison map with visible edge weights."""
    fig, ax = plt.subplots(figsize=(14, 12))
    pos = {node: (node[1], -node[0]) for node in graph.nodes()}

    # 1. Draw the base graph with base weights
    nx.draw(graph, pos, ax=ax, node_size=80, node_color='lightgray',
            with_labels=False, edge_color='gainsboro', style='dotted')
    base_weights = {edge: f"{data['base_weight']:.1f}" for edge, data in graph.edges.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=base_weights, ax=ax, font_color='gray', font_size=7)

    # 2. Draw the routes on top
    legend_handles = [
        Line2D([0], [0], color=data['color'], lw=3, linestyle=data['style'], label=f"{name} ({data['time']:.1f} min)")
        for name, data in routes_data.items()]
    for route_data in routes_data.values():
        path_edges = list(zip(route_data['path'], route_data['path'][1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=route_data['color'],
                               width=route_data['width'], style=route_data['style'], ax=ax)

    # 3. Highlight Start and End nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=[start], node_color='lime', node_size=300, ax=ax, edgecolors='black',
                           label="Start")
    nx.draw_networkx_nodes(graph, pos, nodelist=[end], node_color='red', node_size=300, ax=ax, edgecolors='black',
                           label="End")

    # 4. Add labels, title, and legend
    ax.set_title("Route Comparison Map with Base Travel Weights", fontsize=18, fontweight='bold')
    ax.set_xlabel("City Grid Coordinate (West -> East)", fontsize=14)
    ax.set_ylabel("City Grid Coordinate (North -> South)", fontsize=14)
    ax.legend(handles=legend_handles, loc='upper left', fontsize=12, title="Calculated Routes")

    explanation_text = (
        "This map shows the city road network and the optimal routes found.\n"
        "Gray Numbers: The ideal travel time ('base weight') for each road segment.\n"
        "Colored Lines: The calculated paths for different vehicle types and scenarios."
    )
    # Use fig.text for better placement control relative to the whole figure
    fig.text(0.5, 0.01, explanation_text, ha='center', fontsize=11, bbox=dict(facecolor='aliceblue', alpha=0.8, pad=5))
    fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for the text

    save_plot(fig, filename)


def plot_traffic_heatmap(graph, filename="plot_2_traffic_heatmap.png"):
    """Creates a detailed heatmap of the real-time traffic intensity."""
    fig, ax = plt.subplots(figsize=(10, 9))
    traffic_matrix = np.full((8, 8), 1.0)  # Use 1.0 as the base

    # Average the multiplier for each node based on its connected edges
    node_traffic = {node: [] for node in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        if 'traffic_multiplier' in data:
            val = data['traffic_multiplier']
            node_traffic[u].append(val)
            node_traffic[v].append(val)

    for node, values in node_traffic.items():
        if values:
            traffic_matrix[node[0], node[1]] = np.mean(values)

    im = ax.imshow(traffic_matrix.T, cmap='YlOrRd', interpolation='bilinear', vmin=1, origin='lower')

    ax.set_title("Real-Time Traffic Intensity Heatmap", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("City Grid X-Coordinate", fontsize=14)
    ax.set_ylabel("City Grid Y-Coordinate", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Traffic Multiplier (e.g., 3x travel time)', size=12)

    explanation_text = (
        "This heatmap visualizes live or simulated traffic conditions.\n"
        "Redder areas indicate heavier congestion, which dynamically increases the 'cost' of a path."
    )
    fig.text(0.5, 0.01, explanation_text, ha='center', fontsize=11, bbox=dict(facecolor='beige', alpha=0.6, pad=5))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    save_plot(fig, filename)


def plot_congestion_heatmap(graph, filename="plot_3_congestion_heatmap.png"):
    """Creates a detailed heatmap of the AI's predictive congestion score."""
    fig, ax = plt.subplots(figsize=(10, 9))
    congestion_matrix = np.zeros((8, 8))

    # Average the score for each node based on its connected edges
    node_congestion = {node: [] for node in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        if 'congestion_score' in data:
            val = data['congestion_score']
            node_congestion[u].append(val)
            node_congestion[v].append(val)

    for node, values in node_congestion.items():
        if values:
            congestion_matrix[node[0], node[1]] = np.mean(values)

    im = ax.imshow(congestion_matrix.T, cmap='PuBu', interpolation='bilinear', vmin=0, vmax=1, origin='lower')

    ax.set_title("AI Predictive Congestion Score Heatmap", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("City Grid X-Coordinate", fontsize=14)
    ax.set_ylabel("City Grid Y-Coordinate", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Congestion Score (Probability)', size=12)

    explanation_text = (
        "This heatmap shows the Bayesian Network's prediction of congestion risk.\n"
        "Darker blue signifies a higher probability of delays, allowing proactive routing."
    )
    fig.text(0.5, 0.01, explanation_text, ha='center', fontsize=11, bbox=dict(facecolor='aliceblue', alpha=0.8, pad=5))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    save_plot(fig, filename)


def plot_response_time_comparison(routes_data, filename="plot_4_time_comparison.png"):
    """Creates a bar chart comparing the travel times of different routes."""
    fig, ax = plt.subplots(figsize=(10, 7))
    route_names = list(routes_data.keys())
    times = [routes_data[name]['time'] for name in route_names]
    colors = [routes_data[name]['color'] for name in route_names]

    bars = ax.bar(route_names, times, color=colors, alpha=0.9, edgecolor='black', zorder=3)
    ax.bar_label(bars, fmt='%.1f min', fontsize=11, fontweight='bold', padding=3)

    ax.set_title("Response Time Comparison Across Scenarios", fontsize=18, fontweight='bold')
    ax.set_ylabel("Calculated Travel Time (minutes)", fontsize=14)
    ax.tick_params(axis='x', rotation=10, labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Explanation text is kept minimal as the title and labels are clear.
    explanation_text = "Compares travel times for an emergency vehicle (initial and rerouted)\nversus a regular vehicle in the same traffic conditions."
    fig.text(0.5, 0.01, explanation_text, ha='center', va='bottom', fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    save_plot(fig, filename)


def plot_performance_improvement(routes_data, filename="plot_5_performance.png"):
    """Creates a bar chart visualizing the performance gain of the emergency system."""
    if 'Regular Route' not in routes_data:
        print("Skipping performance plot: Regular route data unavailable.")
        return

    # Use the final rerouted time for the most realistic comparison
    emergency_time = routes_data.get('Rerouted Path', routes_data['Emergency Route'])['time']
    regular_time = routes_data['Regular Route']['time']
    time_saved = regular_time - emergency_time

    if regular_time == 0:
        percentage_saved = 0
    else:
        percentage_saved = (time_saved / regular_time) * 100

    categories = ['Regular Navigation\n(Baseline)', 'AI Emergency System\n(Post-Accident)', 'Time Saved\nby AI System']
    values = [regular_time, emergency_time, time_saved]
    colors_bar = ['#d62728', '#1f77b4', '#2ca02c']  # Red, Blue, Green

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(categories, values, color=colors_bar, alpha=0.9, edgecolor='black', zorder=3)
    ax.bar_label(bars, fmt='%.1f min', fontsize=11, fontweight='bold', padding=3)

    title = f'Emergency System Performance: {percentage_saved:.1f}% Faster Response'
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylabel("Time (minutes)", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.tick_params(axis='x', labelsize=11)

    # A cleaner plot needs less text. The title and labels carry the message.
    fig.tight_layout()
    save_plot(fig, filename)


def plot_statistics_report(routes_data, start, end, evidence, filename="plot_6_statistics.png"):
    """Creates an image file containing a text-based summary of the simulation run."""
    emergency_time_initial = routes_data['Emergency Route']['time']
    regular_time = routes_data['Regular Route']['time']

    final_emergency_time = routes_data.get('Rerouted Path', routes_data['Emergency Route'])['time']
    time_saved = regular_time - final_emergency_time
    if regular_time == 0:
        percentage_saved = 0
    else:
        percentage_saved = (time_saved / regular_time) * 100

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis('off')

    stats_text = f"""
    📊 **SIMULATION ANALYSIS REPORT** 📊

    **SCENARIO PARAMETERS**
    - Origin: {start}   Destination: {end}
    - Time of Day: {evidence['TimeOfDay']}, Weather: {evidence['Weather']}

    ──────────────────────────────────────────────────────────
    **ROUTE ANALYSIS**
    - **Regular Route (Dijkstra):**          {regular_time:.2f} minutes
    - **Initial Emergency Route (A*):**      {emergency_time_initial:.2f} minutes
    """
    if 'Rerouted Path' in routes_data:
        stats_text += f"- **Rerouted Emergency Route (A*):**   **{routes_data['Rerouted Path']['time']:.2f} minutes**"

    stats_text += f"""

    ──────────────────────────────────────────────────────────
    **PERFORMANCE SUMMARY (AI System vs. Regular)**
    - Final AI Route Time:   **{final_emergency_time:.2f} minutes**
    - Time Saved:            **{time_saved:.2f} minutes**
    - Efficiency Improvement: **{percentage_saved:.1f}% faster**
    """

    ax.text(0.02, 0.98, stats_text.replace("    ", ""), transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='azure', alpha=1))

    save_plot(fig, filename)