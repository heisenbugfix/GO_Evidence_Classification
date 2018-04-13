from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support
from preprocessing.preprocess import *
import numpy as np
# For each class
model = load_pkl_data("../data/model_4_13_2018_unique.pickle")
Y_test = load_pkl_data("../data/Ytest_unique.pickle")
X_test = load_pkl_data("../data/Xtest_unique.pickle")
X_test = X_test.tocsc()
y_score = model.predict(X_test)
# pr = precision_recall_fscore_support(Y_test, y_score)
acc = model.score(X_test, Y_test)
print("Accuracy of model is %f"%acc)
precision = dict()
recall = dict()
average_precision = dict()
n_classes = len(evidence_codes)
Y_test = label_binarize(Y_test, classes=np.arange(0, n_classes))
y_score = label_binarize(y_score, classes=np.arange(0, n_classes))
prec = precision_score(Y_test, y_score, labels=np.arange(0, len(evidence_codes)),
                       average='micro')
# for each in prec:
#     print(each)
print("###################")

# print(sum(prec)/len(prec))
# print(prec)
print("Micro averaged precision is %f"%prec)
rec = recall_score(Y_test, y_score, labels=np.arange(0, len(evidence_codes)),
                       average='micro')
# print(len(rec))
print("######################")
# for each in rec:
#     print(each)

f1 = f1_score(Y_test, y_score, labels=np.arange(0, len(evidence_codes)),
              average='micro')
# print(sum(rec)/len(rec))
print("Micro Averaged Recall is %f"%rec)
print("###################")

# print(rec)
print("Micro Averaged f1 is %f"%f1)
# exit(1)
#
#
#
#
#
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
#                                                         y_score[:, i])
#     average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
#
# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
#     y_score.ravel())
# average_precision["micro"] = average_precision_score(Y_test, y_score,
#                                                      average="micro")
# print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#       .format(average_precision["micro"]))

