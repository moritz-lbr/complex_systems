import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Parameters: 10 nodes, probability of edge creation p = 0.3
n = 250
p = 0.01
G = nx.erdos_renyi_graph(n, p)

# Draw the graph
nx.draw(G, with_labels=True, node_color='lightblue', node_size=700, font_size=15)
plt.title(f"Erdős-Rényi Graph with probability of {p} and {n} nodes")
plt.show()


class NetworkAnalysis:
    def __init__(self, graph, bins):
        self.G = graph
        self.bins = bins
    def compute_node_degrees(self):
        """Compute node degrees for all nodes in the graph."""
        self.node_degrees = [self.G.degree(n) for n in self.G.nodes()]
        return self.node_degrees

    def compute_mean_degree(self):
        """Compute the mean degree of the graph."""
        self.compute_node_degrees()  # Ensure node degrees are computed
        mean_degree = sum(self.node_degrees) / len(self.node_degrees) if self.node_degrees else 0
        return mean_degree

    def plot_degree_distribution(self):
        """Plot the distribution of node degrees."""
        self.compute_node_degrees()  # Ensure node degrees are computed
        plt.hist(self.node_degrees, bins=self.bins)
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()

    def compute_local_clustering_coeffs(self):
        """Compute local clustering coefficients for all nodes."""
        self.clustering_coeffs = list(nx.clustering(self.G).values())
        return self.clustering_coeffs

    def compute_mean_clustering_coefficient(self):
        """Compute the mean clustering coefficient of the graph."""
        self.compute_local_clustering_coeffs()  # Ensure clustering coefficients are computed
        mean_clustering = sum(self.clustering_coeffs) / len(self.clustering_coeffs) if self.clustering_coeffs else 0
        return mean_clustering

    def plot_clustering_coefficient_distribution(self):
        """Plot the distribution of clustering coefficients."""
        self.compute_local_clustering_coeffs()  # Ensure clustering coefficients are computed
        clusters, counts = np.unique(self.clustering_coeffs, return_counts=True)
        print(clusters, counts)
        plt.hist(self.clustering_coeffs, bins=self.bins)
        plt.title("Clustering Coefficient Distribution")
        plt.xlabel("Clustering Coefficient")
        plt.ylabel("Frequency")
        plt.show()

    def compute_all_pairs_shortest_path_lengths(self):
        """Compute all pairs' shortest path lengths and return average path lengths for connected components."""
        all_path_lengths = []
        for component in tqdm(nx.connected_components(self.G)):
            subgraph = self.G.subgraph(component)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            all_path_lengths.append(avg_path_length)
        self.all_path_lengths = all_path_lengths
        return self.all_path_lengths

    def compute_mean_path_length(self):
        """Compute the mean path length of the graph."""
        self.compute_all_pairs_shortest_path_lengths()  # Ensure path lengths are computed
        mean_path_length = sum(self.all_path_lengths) / len(self.all_path_lengths) if self.all_path_lengths else 0
        return mean_path_length

    def plot_path_length_distribution(self):
        """Plot the distribution of path lengths."""
        self.compute_all_pairs_shortest_path_lengths()  # Ensure path lengths are computed
        plt.hist(self.all_path_lengths, bins=self.bins, density=True)
        plt.title("Path Length Distribution")
        plt.xlabel("Path Length")
        plt.ylabel("Frequency")
        plt.show()


analysis = NetworkAnalysis(G, 20)
print("Mean Degree:", analysis.compute_mean_degree())
analysis.plot_degree_distribution()
print("Mean Clustering Coefficient:", analysis.compute_mean_clustering_coefficient())
analysis.plot_clustering_coefficient_distribution()
print("Mean Path Length:", analysis.compute_mean_path_length())
analysis.plot_path_length_distribution()





#
# # 1. Compute Node Degrees
# node_degrees = [G.degree(n) for n in G.nodes()]
#
# # 2. Compute Mean Degree
# mean_degree = sum(node_degrees) / len(node_degrees)
# print(f"Mean Degree: {mean_degree}")
#
# # 3. Degree Distribution
# plt.hist(node_degrees, bins=10)
# plt.title("Degree Distribution")
# plt.xlabel("Degree")
# plt.ylabel("Frequency")
# plt.show()
#
#
# # 1. Compute Local Clustering Coefficients
# clustering_coeffs = list(nx.clustering(G).values())
#
# # 2. Compute Mean Clustering Coefficient
# mean_clustering = sum(clustering_coeffs) / len(clustering_coeffs)
# print(f"Mean Clustering Coefficient: {mean_clustering}")
#
# # 3. Clustering Coefficient Distribution
# plt.hist(clustering_coeffs, bins=10)
# plt.title("Clustering Coefficient Distribution")
# plt.xlabel("Clustering Coefficient")
# plt.ylabel("Frequency")
# plt.show()
#
#
#
#
# # 1. Compute All Pairs Shortest Path Lengths
# all_path_lengths = []
#
# for component in tqdm(nx.connected_components(G)):
#     subgraph = G.subgraph(component)
#     avg_path_length = nx.average_shortest_path_length(subgraph)
#     all_path_lengths.append(avg_path_length)
#
# # 2. Compute Mean Path Length
# mean_path_length = sum(all_path_lengths) / len(all_path_lengths) if all_path_lengths else 0
# print(f"Mean Path Length: {mean_path_length}")
# print(all_path_lengths)
# plt.hist(all_path_lengths, bins=10, density=True)
# plt.title("Path Length Distribution")
# plt.xlabel("Path Length")
# plt.ylabel("Frequency")
# plt.show()


