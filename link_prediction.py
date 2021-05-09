import networkx as nx
from tqdm import trange
from fairness_goodness_computation import *
import csv 

Data = open('soc-sign-bitcoinotc.csv', "r")
next(Data, None)  # skip the first line in the input file
Graphtype = nx.DiGraph()

G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype, nodetype=int, data=(('Weight', int),('Timestamp', int)))

# Setting weights -1 to 1
weights = nx.get_edge_attributes(G,"Weight")
for key, val in weights.items():
    weights[key] = val/10
nx.set_edge_attributes(G, weights, "Weight")

print('Computing Fairness and Goodness values')
fairness, goodness = compute_fairness_goodness(G)
nx.set_node_attributes(G,fairness,"fairness")
nx.set_node_attributes(G,goodness,"goodness")

### Calculation of Avg Error in Weight Prediction
avg = 0.0
for edge in G.edges:
    avg+= abs(fairness[edge[0]]*goodness[edge[1]] - weights[edge])
print("Avg Error in Weight Prediction:", avg/len(G.edges))

print(nx.info(G))

# Storing the Network takes too much time! Better to use the F & G values directly to compute edge weight every time!

# G.add_edges_from(list(nx.non_edges(G)))
# dic = {}
# fairness = nx.get_node_attributes(G,"fairness")
# goodness = nx.get_node_attributes(G,"goodness")
# for edge in nx.non_edges(G):
#     dic[edge] = fairness[edge[0]]*goodness[edge[1]]
# nx.set_edge_attributes(G,dic,"Weight")

# nx.write_graphml(G, "updated_graph.graphml", encoding='utf-8', prettyprint=True)