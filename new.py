# import networkx as nx
# G = nx.read_graphml("updated_graph.graphml")
# print(nx.info(G))

import networkx as nx
from tqdm import trange
from fairness_goodness_computation import *
import csv 
import pandas as pd

Data = open('soc-sign-bitcoinotc.csv', "r")
next(Data, None)  # skip the first line in the input file
graph_type = nx.DiGraph()
df = pd.read_csv("soc-sign-bitcoinotc.csv")
print(df.head())
df = df.sort_values(by = "Timestamp")

df.Weight /= 10
split = int(0.6*len(df))
train = df.iloc[:split,:]
test = df.iloc[split:,:]

G = nx.from_pandas_edgelist(train,
                            source='Source',
                            target='Target',
                            edge_attr='Weight',
                            create_using=graph_type)

H = nx.from_pandas_edgelist(test,
                            source='Source',
                            target='Target',
                            edge_attr='Weight',
                            create_using=graph_type)

# G = nx.from_pandas_edgelist(train, "Source", "Target", ["Weight", "Timestamp"])

# H = nx.from_pandas_edgelist(test, "Source", "Target", ["Weight", "Timestamp"])

# G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype, nodetype=int, data=(('Weight', int),('Timestamp', int)))

# Setting weights -1 to 1
# weights = nx.get_edge_attributes(G,"Weight")
# for key, val in weights.items():
#     weights[key] = val/10
# nx.set_edge_attributes(G, weights, "Weight")

print('Computing Fairness and Goodness values of G')
fairness, goodness = compute_fairness_goodness(G)
nx.set_node_attributes(G,fairness,"fairness")
nx.set_node_attributes(G,goodness,"goodness")

### Calculation of Avg Error in Weight Prediction
weights = nx.get_edge_attributes(H,"Weight")
avg = 0.0
count = 0
for edge in H.edges:
    if(G.has_node(edge[0]) and G.has_node(edge[1]) and (not G.has_edge(edge[0],edge[1]))):
        count+=1
        avg+= abs(fairness[edge[0]]*goodness[edge[1]] - weights[edge])
print("Avg Error in Weight Prediction:", avg/count)
print(count)
# print(nx.info(G))
# G.add_edges_from(list(nx.non_edges(G)))
# dic = {}
# fairness = nx.get_node_attributes(G,"fairness")
# goodness = nx.get_node_attributes(G,"goodness")
# for edge in nx.non_edges(G):
#     dic[edge] = fairness[edge[0]]*goodness[edge[1]]
# nx.set_edge_attributes(G,dic,"Weight")

# nx.write_graphml(G, "updated_graph.graphml", encoding='utf-8', prettyprint=True)