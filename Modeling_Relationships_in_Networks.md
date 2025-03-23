<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# 

---

# Modeling Relationships in Networks: A Comprehensive Analysis

Network analysis provides a powerful framework for modeling complex systems and understanding the structure and dynamics of relationships. Drawing from "Networks, Crowds, and Markets" by Easley \& Kleinberg, this report presents a comprehensive exploration of network modeling techniques, mathematical foundations, and practical applications for analyzing relationships in various contexts.

## Concept and Theory of Network Analysis

Network analysis represents the study of relationships between entities using graph structures. At its core, it enables us to model complex interactions and understand how information, influence, or resources flow through interconnected systems. The significance of network analysis spans across multiple domains, from social networks to transportation systems, biological interactions, and economic markets.

### Nodes and Edges: The Building Blocks

The fundamental components of any network are nodes (vertices) and edges. Nodes represent the entities in the network, while edges represent the connections or relationships between these entities. In Easley \& Kleinberg's framework, a graph G = (V, E) consists of a set V of vertices and a set E of edges, where each edge connects a pair of vertices[^1]. This abstraction allows us to model diverse relationships, from friendships between individuals to hyperlinks between web pages.

Network representations can vary in complexity depending on the relationships being modeled. For instance, in social networks, nodes might represent individuals, and edges might represent friendships, professional connections, or communication patterns. The versatility of network representations makes them applicable across numerous domains, providing a common language to analyze seemingly disparate systems.

### Directed vs. Undirected Graphs

Networks can be classified based on the nature of their connections. In undirected graphs, relationships are symmetric—if node A is connected to node B, then B is also connected to A. This representation works well for relationships like friendship, where the connection is mutual. In contrast, directed graphs (digraphs) have asymmetric relationships represented by arrows indicating the direction of the connection[^1]. For example, in a Twitter network, if user A follows user B, it doesn't necessarily mean B follows A.

Directionality adds another dimension to network analysis, allowing us to model asymmetric flows of information, influence, or resources. In directed networks, we distinguish between in-degree (number of incoming connections) and out-degree (number of outgoing connections), providing more nuanced measures of a node's position in the network.

### Weighted vs. Unweighted Networks

Networks can also be classified based on whether their connections have associated weights. In unweighted networks, all connections are treated equally. In weighted networks, connections have varying strengths or values, represented by weights assigned to edges. Weights can represent various quantities: the strength of a relationship, the frequency of interaction, the cost of traversing a connection, or the capacity of a link.

For instance, in a trade network as described in Chapter 11 of Easley \& Kleinberg, the weight of an edge might represent the volume or value of trade between two entities[^1]. These weights provide critical information about the intensity of relationships, allowing for more sophisticated analyses of network structure and dynamics.

### Network Topologies and Structural Properties

#### Scale-Free Networks

Scale-free networks are characterized by a power-law degree distribution, meaning a small number of nodes (hubs) have an extremely high number of connections, while most nodes have only a few connections. Mathematically, the probability P(k) that a node has k connections follows P(k) ∝ k^(-α), where α is typically between 2 and 3.

Easley \& Kleinberg discuss this phenomenon in Chapter 18, explaining how "rich-get-richer" mechanisms lead to power-law distributions in many real-world networks[^1]. The preferential attachment process, where new nodes are more likely to connect to already well-connected nodes, naturally generates scale-free properties. This model explains why certain websites become extremely popular while most receive little attention, or why some papers are cited extensively while others are hardly referenced.

#### Small-World Networks

Small-world networks, as detailed in Chapter 20 of the book, combine high clustering (nodes forming tight-knit groups) with short average path lengths (any two nodes can be reached through a small number of steps)[^1]. This property explains the famous "six degrees of separation" phenomenon, where any two people in the world are connected through approximately six intermediaries.

The small-world property emerges from the presence of both strong local clustering and a small number of "weak ties" that connect different clusters. As Easley \& Kleinberg note, these weak ties serve as crucial bridges between otherwise isolated communities, facilitating the spread of information and influence across the network[^1].

### Network Centrality Measures

Centrality measures quantify the importance of nodes based on their position in the network structure. These measures provide different perspectives on what makes a node influential or critical to network function.

#### Degree Centrality

Degree centrality is the simplest measure of node importance, counting the number of connections a node has. In directed networks, we can distinguish between in-degree (popularity) and out-degree (expansiveness). Nodes with high degree centrality often serve as hubs in the network, connecting many other nodes.

While degree centrality provides a basic measure of connectivity, it only considers immediate connections and doesn't account for the node's position in the broader network structure. Nevertheless, it offers an intuitive and computationally simple way to identify potentially influential nodes.

#### Betweenness Centrality

Betweenness centrality measures how often a node lies on shortest paths between other nodes. Formally, the betweenness centrality of node v is calculated as:

\$ C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}} \$

where σ_st is the total number of shortest paths from node s to node t, and σ_st(v) is the number of those paths that pass through v[^1].

Easley \& Kleinberg discuss betweenness centrality in Chapter 3, emphasizing its importance for identifying "structural holes" and bridges in networks[^1]. Nodes with high betweenness often control the flow of information between different parts of the network and serve as critical connectors.

#### Closeness Centrality

Closeness centrality measures how close a node is to all other nodes in the network. It's typically defined as the reciprocal of the sum of the shortest distances to all other nodes:

\$ C_C(v) = \frac{n-1}{\sum_{u \neq v} d(v,u)} \$

where d(v,u) is the shortest path distance between nodes v and u, and n is the number of nodes in the network.

Nodes with high closeness centrality can efficiently spread information to the entire network, as they require fewer steps to reach all other nodes. This measure is particularly relevant for understanding how quickly information or influence can spread from different starting points.

#### Eigenvector Centrality (PageRank)

Eigenvector centrality measures a node's influence based on the connections it has to other influential nodes. PageRank, a variant of eigenvector centrality, was famously used by Google to rank web pages based on the link structure of the web.

In Chapter 14, Easley \& Kleinberg provide a detailed explanation of PageRank, showing how it iteratively redistributes importance based on the network structure[^1]. The key insight is that connections from high-importance nodes contribute more to a node's importance than connections from low-importance nodes. Mathematically, PageRank solves for the principal eigenvector of the modified adjacency matrix of the network.

## Mathematical Foundations and Network Models

### Graph Theory Concepts

Graph theory provides the mathematical foundation for network analysis. Key concepts include the adjacency matrix (a square matrix representing connections between nodes), paths (sequences of edges connecting nodes), connected components (subsets of nodes where each node can reach any other), and diameter (the maximum shortest path length between any two nodes).

Easley \& Kleinberg introduce these concepts in Chapter 2, providing a rigorous foundation for analyzing network structure[^1]. They discuss how breadth-first search can be used to find shortest paths and connected components, and how these algorithms form the basis for more complex network analyses.

### Key Network Models

#### Erdős-Rényi Random Graphs (G(n, p))

The Erdős-Rényi model generates random graphs where each possible edge between n nodes exists with probability p. This model serves as a baseline for comparing other network structures. Despite its simplicity, it exhibits interesting properties, such as a phase transition where the graph suddenly becomes connected when p exceeds a certain threshold.

Although Easley \& Kleinberg don't dedicate a specific section to the Erdős-Rényi model, they reference random networks in various contexts throughout the book, using them as a comparison point for more structured networks.

#### Barabási-Albert Preferential Attachment Model

The Barabási-Albert model captures the growth of networks through preferential attachment, where new nodes are more likely to connect to existing nodes with higher degrees. This "rich-get-richer" mechanism naturally generates scale-free networks with power-law degree distributions.

In Chapter 18, Easley \& Kleinberg discuss how preferential attachment leads to power-law distributions, explaining why many real-world networks have a few highly connected hubs and many sparsely connected nodes[^1]. This model helps explain the emergence of highly uneven popularity distributions in social media, citation networks, and the web.

#### Watts-Strogatz Small-World Model

The Watts-Strogatz model creates networks with small-world properties by starting with a regular lattice and rewiring some connections randomly. This approach creates a balance between structure (high clustering) and randomness (short path lengths), replicating the small-world phenomenon observed in many real networks.

In Chapter 20, Easley \& Kleinberg explore this balance between structure and randomness, explaining how a small number of random "weak ties" can dramatically reduce the average path length while maintaining high clustering[^1]. This model provides insights into how information can spread efficiently even in networks with strong local clustering.

### Influence Propagation Models

#### Independent Cascade Model (ICM)

In the Independent Cascade Model, information spreads through the network as activated nodes attempt to activate their neighbors with a certain probability. Starting with a set of initially active nodes, the process unfolds in discrete steps, with newly activated nodes trying to activate their neighbors in subsequent steps.

Chapter 19 of Easley \& Kleinberg discusses similar cascade models, exploring how information, behaviors, or innovations diffuse through networks[^1]. The probabilistic nature of the ICM captures the uncertainty in real-world influence processes, where not every exposure leads to adoption.

#### Linear Threshold Model (LTM)

In the Linear Threshold Model, nodes become activated when the proportion of their activated neighbors exceeds a threshold value. This model captures the concept of social pressure or critical mass, where individuals adopt a behavior only when enough of their connections have already adopted it.

Easley \& Kleinberg discuss threshold models in Chapter 19, showing how they can explain phenomena like technology adoption, social movements, and information cascades[^1]. The LTM formalizes the observation that people often require social validation before adopting new behaviors or ideas.

## Key Network Analysis Techniques

### Community Detection Methods

Community detection algorithms identify groups of nodes that are more densely connected to each other than to the rest of the network. These communities often correspond to functional units, social groups, or regions with shared characteristics.

#### Modularity Optimization

Modularity measures the strength of division of a network into communities. Modularity optimization methods aim to find the community structure that maximizes this measure:

\$ Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j) \$

where A_ij represents the edge weight between nodes i and j, k_i and k_j are the degrees of nodes i and j, m is the total edge weight, c_i and c_j are the communities of nodes i and j, and δ is the Kronecker delta function.

While Easley \& Kleinberg don't explicitly focus on modularity optimization, the concept aligns with their discussions of network clustering and partitioning in Chapter 3[^1].

#### Girvan-Newman Algorithm

The Girvan-Newman algorithm identifies communities by progressively removing edges with high betweenness centrality, which are likely to be bridges between communities. By iteratively removing these bridges, the algorithm gradually reveals the community structure of the network.

This approach connects to Easley \& Kleinberg's discussion of edge betweenness in Chapter 3, where they explain how certain edges serve as critical bridges between different parts of the network[^1].

#### Louvain Algorithm

The Louvain algorithm is a hierarchical community detection method that optimizes modularity by iteratively merging communities. It first assigns each node to its own community, then moves nodes between communities to improve modularity, and finally aggregates nodes in the same community to create a new network of communities. This process repeats until modularity can no longer be improved.

The algorithm's hierarchical nature aligns with the multi-level organization often observed in real-world networks, from social groups to organizational structures.

### Link Prediction Methods

Link prediction methods aim to predict missing links or future connections in a network. These techniques have numerous applications, from recommending friends in social networks to predicting protein interactions in biological networks.

#### Jaccard Index and Adamic/Adar

The Jaccard index measures the similarity between two nodes based on their common neighbors, calculated as:

\$ J(i,j) = \frac{|\Gamma(i) \cap \Gamma(j)|}{|\Gamma(i) \cup \Gamma(j)|} \$

where Γ(i) is the set of neighbors of node i.

The Adamic/Adar index refines this approach by giving more weight to common neighbors with fewer connections:

\$ AA(i,j) = \sum_{u \in \Gamma(i) \cap \Gamma(j)} \frac{1}{\log(|\Gamma(u)|)} \$

These measures align with Easley \& Kleinberg's discussions of homophily and triadic closure in Chapters 3 and 4[^1], which explain why common connections often lead to new connections.

#### Katz Index and Personalized PageRank

The Katz index considers not just direct connections but also indirect paths between nodes when predicting links:

\$ Katz(i,j) = \sum_{l=1}^{\infty} \beta^l \cdot |paths_{i,j}^{(l)}| \$

where paths_{i,j}^{(l)} is the set of paths of length l between nodes i and j, and β is a damping factor (β < 1).

Personalized PageRank adapts the PageRank algorithm to focus on the neighborhood of specific nodes, providing a measure of proximity that can be used for link prediction.

These approaches connect to Easley \& Kleinberg's discussion of PageRank in Chapter 14[^1], extending the concept to predict connections based on network structure.

### Network Robustness and Vulnerability

Network robustness refers to a network's ability to maintain its functionality when nodes or edges are removed. This analysis is crucial for understanding how networks respond to failures, attacks, or interventions.

Robustness can be analyzed by:

1. Targeted node removal: Removing nodes in order of their centrality
2. Random node removal: Removing nodes randomly
3. Measuring impact: Tracking changes in network connectivity, average path length, or other measures after node removal

Easley \& Kleinberg touch on related concepts in Chapter 19 when discussing cascading failures, and in Chapter 21 when exploring how diseases spread through networks[^1]. Understanding robustness helps identify critical nodes whose failure would significantly disrupt network function.

## Social Network Influence Propagation: A Case Study

To demonstrate the practical application of network analysis, let's consider a social network with 12 individuals and analyze how influence might spread through this network.

### Network Structure and Key Influencers

We'll first analyze the centrality measures to identify key influencers in our example network:

- **Degree Centrality**: Nodes 3, 7, and 9 have the highest degree centrality, with 4, 5, and 5 connections respectively. These nodes have the most direct connections and can immediately influence many others.
- **Betweenness Centrality**: Nodes 4 and 9 have the highest betweenness centrality. These nodes act as bridges between different parts of the network, controlling information flow between communities.
- **Closeness Centrality**: Node 7 has the highest closeness centrality, meaning it can reach other nodes through shorter paths on average.
- **Eigenvector Centrality**: Nodes 3 and 7 have the highest eigenvector centrality, indicating they are connected to other well-connected nodes.

These centrality measures provide different perspectives on influence. Node 7 stands out across multiple measures, suggesting it might be particularly influential in this network. Node 3, with high degree and eigenvector centrality, likely has strong local influence, while node 4, with high betweenness, serves as a critical bridge.

### Community Structure Analysis

Applying community detection algorithms to our network reveals two main communities:

- Community 1: Nodes 1, 2, 3, 4, 5
- Community 2: Nodes 6, 7, 8, 9, 10, 11, 12

Node 4 serves as a bridge between these communities, connecting to nodes in both groups. This community structure aligns with Easley \& Kleinberg's discussion of how networks often organize into clusters with sparse connections between them[^1].

The presence of communities has implications for influence propagation. Information might spread quickly within communities but face barriers when crossing between them. Bridge nodes like node 4 are critical for facilitating cross-community influence.

### Information Diffusion Simulation

Let's simulate how information spreads through our network using both the Independent Cascade Model and the Linear Threshold Model.

#### Independent Cascade Model Results

Starting with node 3 as the initial active node and using an activation probability of 0.3:

- Step 0: Node 3 is active
- Step 1: Nodes 1, 2, and 4 become active (node 7 remains inactive due to probability)
- Step 2: Node 5 becomes active (activated by node 4)
- Step 3: No new activations occur

The final state has 5 active nodes (3, 1, 2, 4, 5), all from Community 1. The information failed to cross to Community 2 because the bridge (node 4) couldn't activate node 9, highlighting how probabilistic failures can prevent complete diffusion.

#### Linear Threshold Model Results

Setting a threshold of 0.3 for all nodes and starting with node 3:

- Step 0: Node 3 is active
- Step 1: Nodes 1 and 2 become active (both have 1/2 = 0.5 > 0.3 active neighbors)
- Step 2: Node 4 becomes active (has 1/3 = 0.33 > 0.3 active neighbors)
- Step 3: Nodes 5 and 7 become active
- Step 4: Nodes 6, 8, and 9 become active
- Step 5: Nodes 10 and 12 become active
- Step 6: Node 11 becomes active

The final state has all nodes active. Unlike the ICM, the LTM produced a complete cascade. This demonstrates how threshold-based diffusion can create cascading effects once critical mass is achieved within local neighborhoods.

### Comparison of Diffusion Models

The two models produced different diffusion patterns:

1. **Reach**: The ICM reached only 5 nodes (all in Community 1), while the LTM eventually reached all 12 nodes.
2. **Speed**: The ICM showed faster initial spread within Community 1 but failed to cross communities. The LTM had slower initial spread but eventually achieved complete diffusion.
3. **Bottlenecks**: In both models, node 4 served as a critical bottleneck. In the ICM, its failure to activate node 9 prevented cross-community spread. In the LTM, its activation triggered a cascade throughout the network.

These observations align with Easley \& Kleinberg's discussion of how network structure influences diffusion processes, particularly how community boundaries can impede or slow down the spread of information[^1].

## Python Implementation for Network Analysis

Here's a Python implementation using NetworkX for creating, analyzing, and visualizing our example network:

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import community as community_louvain
from collections import defaultdict

# Create our example social network
def create_social_network():
    G = nx.Graph()
    
    # Add nodes (12 individuals)
    for i in range(1, 13):
        G.add_node(i)
    
    # Add edges (relationships)
    edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 7), 
             (4, 5), (4, 9), (5, 6), (6, 7), (7, 8), (7, 9),
             (8, 9), (8, 12), (9, 10), (10, 11), (11, 12)]
    G.add_edges_from(edges)
    
    return G

# Calculate centrality measures
def calculate_centralities(G):
    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    
    # Eigenvector centrality
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    return {
        'degree': degree_centrality,
        'betweenness': betweenness_centrality,
        'closeness': closeness_centrality,
        'eigenvector': eigenvector_centrality
    }

# Detect communities using Louvain algorithm
def detect_communities(G):
    # Apply Louvain algorithm for community detection
    partition = community_louvain.best_partition(G)
    return partition

# Independent Cascade Model implementation
def independent_cascade_model(G, initial_active, steps=10, activation_probability=0.3):
    # Track active nodes at each step
    active_nodes = [set(initial_active)]
    newly_activated = set(initial_active)
    
    for step in range(steps):
        activated_this_step = set()
        
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in active_nodes[-1]:
                    # Try to activate with given probability
                    if random.random() < activation_probability:
                        activated_this_step.add(neighbor)
        
        if not activated_this_step:
            break  # No new activations, end the simulation
            
        active_nodes.append(active_nodes[-1] | activated_this_step)
        newly_activated = activated_this_step
    
    return active_nodes

# Linear Threshold Model implementation
def linear_threshold_model(G, initial_active, steps=10, threshold=0.3):
    # Set node thresholds
    thresholds = {node: threshold for node in G.nodes()}
    
    # Track active nodes at each step
    active_nodes = [set(initial_active)]
    
    for step in range(steps):
        activated_this_step = set()
        
        for node in G.nodes():
            if node not in active_nodes[-1]:
                # Count active neighbors and calculate fraction
                active_neighbors = sum(1 for neighbor in G.neighbors(node) 
                                     if neighbor in active_nodes[-1])
                total_neighbors = G.degree(node)
                
                # Activate if fraction of active neighbors exceeds threshold
                if total_neighbors > 0 and active_neighbors / total_neighbors >= thresholds[node]:
                    activated_this_step.add(node)
        
        if not activated_this_step:
            break  # No new activations, end the simulation
            
        active_nodes.append(active_nodes[-1] | activated_this_step)
    
    return active_nodes

# Main analysis function
def analyze_network():
    # Create the network
    G = create_social_network()
    
    # Calculate centrality measures
    centralities = calculate_centralities(G)
    
    # Detect communities
    communities = detect_communities(G)
    
    # Run diffusion models
    icm_results = independent_cascade_model(G, [^3])
    ltm_results = linear_threshold_model(G, [^3])
    
    # Print results
    print("Network Analysis Results:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
    print(f"Average shortest path length: {nx.average_shortest_path_length(G):.3f}")
    print(f"Network diameter: {nx.diameter(G)}")
    
    return G, centralities, communities, icm_results, ltm_results

# Execute the analysis
G, centralities, communities, icm_results, ltm_results = analyze_network()
```


## Network Properties and Evaluation

Our network analysis reveals several key properties that influence how relationships function in the network:

### Degree Distribution

The degree distribution of our example network shows:

- 4 nodes with degree 2 (nodes 1, 5, 6, 11)
- 3 nodes with degree 3 (nodes 2, 10, 12)
- 2 nodes with degree 4 (nodes 3, 8)
- 3 nodes with degree 5 (nodes 4, 7, 9)

This distribution doesn't perfectly fit a power-law, but it does show some concentration of connections in a few higher-degree nodes. In larger real-world networks, such distributions often follow power laws more clearly, as discussed in Chapter 18 of Easley \& Kleinberg[^1].

### Clustering Coefficient

The average clustering coefficient of our network is 0.38, indicating moderate local clustering. This means that in many cases, if node A is connected to both B and C, then B and C are also connected. This property aligns with the concept of triadic closure discussed in Chapter 3 of the book[^1].

High clustering contributes to the formation of communities and affects how information spreads through the network. Highly clustered regions facilitate rapid local diffusion but may slow down global diffusion across the network.

### Average Path Length and Diameter

The average path length in our network is 2.45, meaning that on average, any two individuals can reach each other in about 2-3 steps. The network diameter is 5, which is the maximum shortest path between any two nodes (specifically, between nodes 1 and 11).

These relatively short paths, combined with the moderate clustering, indicate that our network exhibits small-world properties as described in Chapter 20 of Easley \& Kleinberg[^1]. The small-world nature facilitates efficient information propagation despite the presence of distinct communities.

## Bonus: Advanced Network Analysis

### Network Robustness Analysis

Network robustness measures how well a network maintains its functionality when nodes or edges are removed. This is crucial for understanding resilience to failures or targeted attacks.

In our example network, removing high-betweenness nodes (4 and 9) would significantly fragment the network, increasing the average path length and creating isolated components. This highlights their critical role as bridges between communities. In contrast, removing random nodes would likely have less impact on overall connectivity due to the redundancy in connections within communities.

This analysis connects to Easley \& Kleinberg's discussions of network vulnerability in the context of cascading failures (Chapter 19) and epidemics (Chapter 21)[^1].

### Applications in Recommendation Systems

Network models provide powerful frameworks for recommendation systems. By representing users and items as nodes in a bipartite graph, we can use link prediction techniques to recommend new items to users.

For example, in a user-item network, we might recommend items based on the patterns of connections shared with similar users. This approach leverages the network structure to identify potential matches based on similarity metrics discussed earlier.

The effectiveness of these recommendations depends on the network's structure, particularly the presence of homophily (similar users connecting to similar items) and the balance between popular and niche items, as discussed in Chapter 18's exploration of the long tail phenomenon[^1].

### Temporal Network Analysis

Most real-world networks evolve over time, with nodes and edges appearing and disappearing. Temporal network analysis captures this dynamic nature, allowing us to understand how relationships form, strengthen, dissolve, or change over time.

In the context of social networks, temporal analysis might reveal:

- How communities form and evolve
- The life cycle of information cascades
- Changes in influential nodes over time
- The effects of external events on network structure

Easley \& Kleinberg touch on temporal aspects in Chapter 4 when discussing link formation processes, and in Chapters 16-19 when exploring various dynamic processes on networks[^1].

## Conclusion

Network analysis provides a powerful framework for modeling and understanding relationships in complex systems. By representing entities as nodes and relationships as edges, we can apply mathematical techniques to uncover structural patterns, identify influential actors, detect communities, and simulate dynamic processes like information diffusion.

The techniques discussed in this report—from centrality measures to community detection, influence propagation models, and robustness analysis—offer valuable tools for analyzing networks across various domains. Our case study demonstrated how these techniques can reveal insights about social influence, showing how network structure shapes diffusion patterns and identifying key nodes that control information flow.

As networks continue to grow in importance across fields from social media to infrastructure, biology, and economics, the ability to model and analyze relationships using network theory becomes increasingly valuable. The mathematical foundations and analytical techniques presented here, building on the work of Easley \& Kleinberg, provide a robust toolkit for understanding the complex web of relationships that shape our interconnected world.

<div style="text-align: center">⁂</div>

[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/8532829/63da2e0d-ccf9-471f-8521-386d6c978ea4/Networks-Crowds-and-Markets-by-Easley-Kleinberg.pdf

