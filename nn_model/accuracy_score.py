import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class model_evaluation:

	def __init__(self,y_true,y_pred):
		
		self.num_samples = y_true.shape[0]
		self.num_class = y_true.shape[1]

	def compute_accuracy_score(self, y_true, y_pred):
		val = np.zeros((self.num_samples,self.num_class))
		val[y_true == y_pred] = 1
		val[y_true == 0] = 0
		accuracy_score =  np.sum(val, axis=1)/self.num_class 
		print(accuracy_score)
		
		return accuracy_score

	def binary_class_model(self, y_true, y_pred ):

		f1_mac =[]
		f1_mic = []
		f1_weighted = []
		precision= []
		recall= []

		for i in range(self.num_class):
			y_true_binary = y_true[:,i]
			y_pred_binary = y_pred[:,i]
			f1_mac.append(f1_score(y_true_binary, y_pred_binary , average='macro'))
			f1_mic.append(f1_score(y_true_binary, y_pred_binary , average='micro'))
			f1_weighted.append(f1_score(y_true_binary, y_pred_binary , average='weighted'))
			precision.append(precision_score(y_true_binary, y_pred_binary , average='weighted'))
			recall.append(recall_score(y_true_binary, y_pred_binary, average='weighted'))

		return f1_mac, f1_mic, f1_weighted, precision, recall


'''
path = "/Users/nehachoudhary/Documents/Oracle_Lab"
with open("label_example.pickle", 'rb') as f:
        label = pickle.load(f)
	
y_true = label[0] 
y_pred = label[1] 
'''

#calling Model Eval

'''
eval = model_evaluation(y_true,y_pred)

accuracy_score = eval.compute_accuracy_score(y_true, y_pred)
f1_mac, f1_mic, f1_weighted, precision, recall = eval.binary_class_model(y_true, y_pred)

Model_metrics = [f1_mac, f1_mic, f1_weighted, precision, recall]


with open('accuracy_score.pickle', 'wb') as handle:
	pickle.dump(accuracy_score, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Model_metrics.pickle', 'wb') as handle:
	pickle.dump(Model_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''


