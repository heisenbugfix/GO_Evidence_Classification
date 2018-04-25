from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from preprocessing.preprocess import *
import numpy as np
# For each class
model = load_pkl_data("../data/model_4_13_2018.pickle")
Y_test = load_pkl_data("../data/Ytest.pickle")
X_test = load_pkl_data("../data/Xtest.pickle")
X_test = X_test.tocsc()
y_score = model.predict(X_test)
scores = []
scores.append(Y_test)
scores.append(y_score)
save_pkl_data("logistic_regression_test_scores.pickle", scores)
exit(1)
acc = model.score(X_test, Y_test)
print("Accuracy of model is %f"%acc)
precision = dict()
recall = dict()
average_precision = dict()
n_classes = len(evidence_codes)
Y_test = label_binarize(Y_test, classes=np.arange(0, n_classes))
y_score = label_binarize(y_score, classes=np.arange(0, n_classes))

print("###################")
prec = precision_score(Y_test, y_score, labels=np.arange(0, len(evidence_codes)),
                       average="weighted")
print("Weighted precision is %f"%prec)

print("######################")
rec = recall_score(Y_test, y_score, labels=np.arange(0, len(evidence_codes)),
                       average='weighted')
print("Weighted Recall is %f"%rec)

print("######################")
f1 = f1_score(Y_test, y_score, labels=np.arange(0, len(evidence_codes)),
              average='weighted')
print("Weighted Averaged f1 is %f"%f1)

print("###################")
f2 = f1_score(Y_test, y_score, labels=np.arange(0, len(evidence_codes)),
              average='macro')
print("Macro Averaged f1 is %f"%f2)

print("###################")
f3 = f1_score(Y_test, y_score, labels=np.arange(0, len(evidence_codes)),
              average='micro')
print("Micro Averaged f1 is %f"%f3)

print(f1)
print(f2)
print(f3)
print(prec)
print(rec)

