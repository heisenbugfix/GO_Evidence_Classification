import pickle as pkl
import json

labels = ["DB","DB_OID","DB_OBS",
          "Qualifier","GO_ID","DB_REF","EVIDENCE",
          "WITH","Aspect","DB_OBN","DB_OBSYN",
          "DB_OBType","Taxon","Date", "Assigned_By",
          "Annotation_EXT","Gene_PFID"]

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

data = load_json_data("../data/all_data.json")
print("OK")
