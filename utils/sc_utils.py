# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt

# def generate_random_graph(num_clients, edge_probability=0.15, seed=None):
#     """Generate a random connected graph using Erdős-Rényi model"""
#     while True:
#         g = nx.erdos_renyi_graph(num_clients, edge_probability, seed=seed)
#         if nx.is_connected(g):
#             return g

# def generate_graph_by_type(num_clients, graph_type='erdos_renyi', **kwargs):
#     """Generate different types of graphs for experiments"""
#     if graph_type == 'erdos_renyi':
#         p = kwargs.get('edge_probability', 0.15)
#         return generate_random_graph(num_clients, p, kwargs.get('seed', None))
    
#     elif graph_type == 'small_world':
#         k = kwargs.get('k', 4)  # Each node connected to k nearest neighbors
#         p = kwargs.get('p', 0.1)  # Rewiring probability
#         return nx.watts_strogatz_graph(num_clients, k, p, seed=kwargs.get('seed', None))
    
#     elif graph_type == 'scale_free':
#         m = kwargs.get('m', 2)  # Number of edges to attach from new node
#         return nx.barabasi_albert_graph(num_clients, m, seed=kwargs.get('seed', None))
    
#     elif graph_type == 'complete':
#         return nx.complete_graph(num_clients)
    
#     else:
#         raise ValueError(f"Unknown graph type: {graph_type}")

# def visualize_graph(graph, title="Communication Graph", save_path=None):
#     """Visualize the communication graph"""
#     plt.figure(figsize=(10, 8))
#     pos = nx.spring_layout(graph, seed=42)
#     nx.draw(graph, pos, with_labels=True, node_color='skyblue', 
#             node_size=700, edge_color='gray', width=2)
#     plt.title(title, fontsize=16)
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.show()

# def get_graph_statistics(graph):
#     """Get statistics about the graph"""
#     stats = {
#         'num_nodes': graph.number_of_nodes(),
#         'num_edges': graph.number_of_edges(),
#         'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
#         'max_degree': max(dict(graph.degree()).values()),
#         'min_degree': min(dict(graph.degree()).values()),
#         'diameter': nx.diameter(graph) if nx.is_connected(graph) else float('inf'),
#         'avg_clustering': nx.average_clustering(graph),
#         'is_connected': nx.is_connected(graph)
#     }
#     return stats
