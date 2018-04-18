import pickle as pkl
import numpy as np
from preprocessing.preprocess import *
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
data_vecs = load_pkl_data("../data/Xtrain_unique.pickle")
data_vecs = data_vecs.tocsc()
data_labels = load_pkl_data("../data/Ytrain_unique.pickle")

# train_len = int(data_vecs.shape[0] * 0.8)
X_train = data_vecs
X_test = load_pkl_data("../data/Xtest_unique.pickle")
X_test = X_test.tocsc()
Y_train = data_labels
Y_test = load_pkl_data("../data/Ytest_unique.pickle")
model = LogisticRegression( )
model.fit(X_train, Y_train)
save_pkl_data("../data/model_4_13_2018_unique.pickle",model)
print (model.score(X_test, Y_test))