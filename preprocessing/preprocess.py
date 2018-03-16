import pickle as pkl
import obonet
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

labels = ["DB","DB_OID","DB_OBS",
          "Qualifier","GO_ID","DB_REF","EVIDENCE",
          "WITH","Aspect","DB_OBN","DB_OBSYN",
          "DB_OBType","Taxon","Date", "Assigned_By",
          "Annotation_EXT","Gene_PFID"]

feature_labels = {"GO_ID", "DB_REF", "GO_PARENTS", "Aspect", "Taxon", "DB_OBSYN", "WITH"}

def load_pkl_data(filename):
    f = open(filename, 'rb')
    data = pkl.load(f)
    return data

def load_json_data(filename):
    f = open(filename)
    data = json.load(f)
    return data

def dump_to_json(filename, outfile=None):
    # dat = load_data("../data/new_annotations.pkl")
    dat = load_pkl_data(filename)
    data_dict = []
    for each in dat:
        curr = {}
        for val, label in zip(each, labels):
            curr[label] = val
        data_dict.append(curr)
    if outfile is None:
        with open("../data/all_data.json",'w') as f:
            json.dump(data_dict, f)
    else:
        with open(outfile,'w') as f:
            json.dump(data_dict, f)

def load_pubmed_text_data(filename=None):
    if not filename:
        filename = "../data/pubmed_output.txt"
    data = {}
    with open(filename, 'r', encoding='latin1')as f:
        for each in f:
            splitted = each.split('\t',2)
            pubid = splitted[0]
            pubyr = splitted[1]
            abstract = splitted[2]
            data[pubid] = [{"date": pubyr, "abstract": abstract}]
    return data

def dump_pubmed_json_fromtext(infile=None, outfile=None):
    data = load_pubmed_text_data(infile)
    if not outfile:
        outfile = "../data/pubmed.json"
        with open(outfile, 'w') as f:
            json.dump(data,f, indent=2)

def temp():
    corpus = ["This is very strange palce in a strange land for it to be true with building car ghost town hood jacket",
              "This is very nice"]
    cvec = CountVectorizer()
    Y = cvec.fit_transform(corpus)
    tf_transformer = TfidfTransformer()
    X_train_tfidf = tf_transformer.fit_transform(Y)
    a = np.asarray(X_train_tfidf[1].todense()).flatten()
    print(a)
    print("OK")

def create_go_term_vector(filename=None, dump=False):
    if not filename:
        filename = "../data/go.obo"
    graph = obonet.read_obo(filename)
    nodes = graph._adj.keys()
    node_to_index = {}
    index_to_node = {}
    for i, node in enumerate(nodes):
        node_to_index[node] = i
        index_to_node[i] = node
    # ohenc = OneHotEncoder(n_values=len(node_to_index))
    # data = ohenc.fit([[1],[3456],[234],[4367]])
    if dump:
        with open("../data/node_to_index.pkl", 'wb') as f:
            pkl.dump(node_to_index, f)
        with open("../data/index_to_node.pkl", 'wb') as f:
            pkl.dump(index_to_node, f)
    return node_to_index, index_to_node

def get_parent_nodes(node, graph=None):
    if not graph:
        graph = obonet.read_obo("../data/go.obo")
    ans = []
    node_dic = graph._adj[node]
    for every in node_dic:
        for every_key in node_dic[every]:
            if every_key == "is_a":
                ans.append(every)
    return ans




