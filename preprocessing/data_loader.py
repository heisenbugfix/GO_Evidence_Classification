from __future__ import division
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
go_id = set()
i = 0
aspects = set()
with1 = set()
synonym = set()
syn = {}
taxon = set()
db_otype = set()
for en, each in enumerate(x):
    aspects.add(each["Aspect"])
    for val in each['With']:
        with1.add(val)
    for val in each['Synonym']:
        synonym.add(val)
        if val in syn:
            syn[val]+=1
        else:
            syn[val] = 1
    for val in each['Taxon_ID']:
        taxon.add(val)
    db_otype.add(each["DB_Object_Type"])
    if each["DB_Object_Type"]!='protein':
        print(each["DB_Object_Type"])
        i += 1
    for every in each["DB:Reference"]:
        if "GO_REF" in every:
            go_id.add(every)

print(en)
print(len(go_id))
print(i)
print("ASPECTS")
print(aspects)
print("WITH")
print(with1)
print("SYNONYM")
sum = 0
for each in syn.values():
    sum+=each
sum = sum/len(syn)
print("average : %f"%sum)
print(len(synonym))
print("TAXON_ID")
print(taxon)
print("DB_OBJECT_TYPES")
print(db_otype)
# find_statistics()

