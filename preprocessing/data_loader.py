import pickle as pkl
from Bio.UniProt.GOA import _gaf20iterator

def load_order_embedding_data():
    with open("negative_training_examples.pkl",'rb') as ptr:
        neg_ex = pkl.load(ptr)

    with open("positive_training_examples.pkl",'rb') as ptr:
        pos_ex = pkl.load(ptr)

def find_statistics(filepath=None):
    if filepath is None:
        f = open("../data/alldb.txt")
    else:
        f = open(filepath)
    for each in f:
        sp_data = each.split("  ")

x = _gaf20iterator(open("../data/goa_uniprot_all_noiea.gaf", 'r'))
go_id = []
for each in x:
    for every in each["DB:Reference"]:
        if "PMID" in every :
            go_id.append(each)
print(len(go_id))
# find_statistics()

