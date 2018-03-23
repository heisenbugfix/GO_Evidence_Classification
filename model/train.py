import pickle as pkl
import numpy as np
from preprocessing.preprocess import load_pkl_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
evidence_codes = ["EXP", "IDA", "IPI",
                  "IMP", "IGI", "IEP",
                  "HTP", "HDA", "HMP",
                  "HGI", "HEP",
                  "ISS", "ISO", "ISA",
                  "ISM", "IGC", "IBA",
                  "IBD", "IKR", "IRD",
                  "RCA", "TAS", "NAS",
                  "IC", "ND"]
data_vecs = load_pkl_data("../data/data_feature_vector.pickle")
data_vecs = data_vecs.tocsc()
data_labels = load_pkl_data("../data/data_labels_vector_non_sparse.pickle")

# Y = label_binarize(data_labels, classes=np.arange(0, len(evidence_codes)))
# data_labels = data_labels.tocsc()
# X_train, X_test, Y_train, Y_test  = train_test_split(data_vecs, Y, test_size=0.2)
train_len = int(data_vecs.shape[0] * 0.8)
X_train = data_vecs[0: train_len]
X_test = data_vecs[train_len: ]
Y_train = data_labels[0: train_len]
Y_test = data_labels[train_len:]

model = LogisticRegression( )
model.fit(X_train, Y_train)
with open("../data/model.pickle",'wb') as f:
    pkl.dump(model, f)
print (model.score(X_test, Y_test))