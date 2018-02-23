import obonet
import random
import pickle as pkl

graph = obonet.read_obo("go.obo")
positive_pairs = []
keys1 = []
print(graph._adj.keys())
for each in graph._adj:
    if each=='GO:0000072':
        print("OK")
    if "is_obsolete" in graph._node[each]:
        if graph._node[each]["is_obsolete"]:
            continue
    keys1.append(each)
    node_dic = graph._adj[each]
    for every in node_dic:
        for every_key in node_dic[every]:
            if every_key=="is_a":
                positive_pairs.append((each, every))

keys2 = list(keys1)
random.shuffle(keys1)
negative_pairs = []
i = 0
while i < 400000:
    idx1 = random.randint(0, len(keys1)-1)
    idx2 = random.randint(0, len(keys2)-1)
    node1 = keys1[idx1]
    node2 = keys2[idx2]
    if not graph.has_successor(node1, node2) and not graph.has_predecessor(node1, node2):
        negative_pairs.append((node1, node2))
        i = i + 1
with open("positive_training_examples.pkl",'wb') as ptr:
    pkl.dump(positive_pairs,ptr)

with open("negative_training_examples.pkl",'wb') as ptr:
    pkl.dump(negative_pairs,ptr)


