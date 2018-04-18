import os
import csv
import math
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import all_go_data_pre as all_go
import operator
import itertools
from collections import defaultdict
import io
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
import pickle as pkl

evidence_codes = ["EXP", "IDA", "IPI",
                  "IMP", "IGI", "IEP",
                  "HTP", "HDA", "HMP",
                  "HGI", "HEP","ISS",
                  "ISO", "ISA","ISM", 
                  "IGC", "IBA","IBD",
                  "IKR", "IRD","RCA",
                  "TAS", "NAS","IC", 
                  "ND"]                

def get_data(data_file):
	params = all_go.read_json_input(data_file)
	evidence_dict = all_go.get_evidence_codes(params)
	d1 = {i:j['code'] for i,j in evidence_dict.items()}
	#print(d1)
	pubmed_data = all_go.read_json_input("pubmed_latest.json")
	d2 = {i:j['abstract'] for i,j in pubmed_data.items()}
	#print(d2)
	model_data = defaultdict(list)
	for d in (d1, d2):
    		for key, value in d.items():
        		model_data[key].append(value)
	model_data_2 = {key:value for key, value in model_data.items() if len(value)==2}
	model_data_3 = {}
	for key, value in model_data.items():
    		if len(value)==2:
        		model_data_3.setdefault(value[0], []).append(value[1])
	return model_data_3, pubmed_data

#get a dictionary of characters
def get_chars (pubmed_data):
	kv={}
	abstract=[]
	for c in pubmed_data.values():
    		abstract.append(c['abstract'])
    		for k in c['abstract']:
        		kv[k]=kv.setdefault(k, 0)+1

	test=''.join(kv.keys())
	test=''.join(sorted(test))
	
	#filter characters with frequency of atleast 10
	kv1={k: v for k, v in kv.items() if v >500}
	kv2={k: v for k, v in kv.items() if v <501}

	#token for characters with low freq
	#start stop tokens
	kv1['<Other>']=sum(kv2.values())
	kv1['<SS>']=len(abstract)
	kv1['</S>']=len(abstract)
	return kv1, kv2

#char2int
def alphaint(dix):
    word2int={}
    int2word={}
    for i,k in enumerate(dix.keys()):
        word2int[k] = i
        int2word[i] = k
    return word2int, int2word


#creating dictionary of y - embed code
#need to edit after safai of model data 2
def y_embed_code(model_data_3):
	y_dict={}
	for i,j in model_data_3.items():
    		y_dict[i]=len(j)
	return y_dict


###Unigram Model (1st order Markov model): 
###for every y compute aij
###numerator and denominator are calculated separately for efficient computing
def unigram_model(data,kv1,kv1_word2int):
	Nall=defaultdict(list)
	Dall=defaultdict(list)
	count_all=0
	print(data.keys())
	for l in data.keys():
    		N=np.zeros(shape=(len(kv1),len(kv1)))
    		D=np.zeros(shape=(len(kv1)))
    		count=0
    		for t in data[l]:
        		x = [j if j in kv1.keys() else '<Other>' for j in t]
        		N[kv1_word2int['<SS>']][kv1_word2int[x[0]]]+=1
        		D[kv1_word2int['<SS>']]+=1
        		if len(x)-1 > 0:
            			for k in range(len(x)-1):
                			N[kv1_word2int[x[k]]][kv1_word2int[x[k+1]]]+=1
                			D[kv1_word2int[x[k]]]+=1
            			N[kv1_word2int[x[k+1]]][kv1_word2int['</S>']]+=1
				D[kv1_word2int[x[k+1]]]+=1
        		else:
            			N[kv1_word2int[x[0]]][kv1_word2int['</S>']]+=1
            			D[kv1_word2int[x[0]]]+=1
        		count+=1
        		count_all+=1
		print("evd code:", l, "abs in evd code:", count, "abs_all", count_all)
		Nall[l]=N
		Dall[l]=D
	return Nall, Dall

def laplace_smoothing(Nall, Dall):
	### laplacian smoothing for characters with 0 freq
	Aall_smooth=defaultdict(list)
	for l in data.keys(): Aall_smooth[l] = (Nall[l]+1)/(Dall[l]+len(kv1))
        return Aall_smooth

def compute_start_state(data,kv1):
	Piall=defaultdict(list)
	count_all=0
	#x = list(data[ylang.keys()[0]][0])
	for l in data.keys():
		Pi=np.zeros(shape=(len(kv1)))
		count=0
		for t in data[l]:
			j=t[0]
        		if j in kv1.keys(): x = j
        		else: x = '<Other>'
			Pi[kv1_word2int[x]]+=1
        		#print("evd code:", l, "abs in evd code:", count, "abs_all", count_all)
			count+=1
			count_all+=1
		print("evd code:", l, "abs in evd code:", count, "abs_all", count_all)
		Piall[l]=(Pi+1)/(len(data[l])+len(kv1))
	return Piall

def prediction(data, Aall_smooth, kv1,Piall, kv1_word2int):
	#predict y on train/test data
	pred_y = []
	val_y = []
	count=0
	correct=0
	for l in data.keys():
    		for t in data[l]:
        		x = [j if j in kv1.keys() else '<Other>' for j in t]
			Pi_val=np.log(np.array([Piall[ll][kv1_word2int[x[0]]] for ll in data.keys()]))
        		A_val=Pi_val+np.log(np.array([Aall_smooth[ll][kv1_word2int['<SS>']][kv1_word2int[x[0]]] for ll in data.keys()]))
			if len(x)-1 > 0:
				for k in range(len(x)-1):
                			A_val = A_val+np.log(np.array([Aall_smooth[ll][kv1_word2int[x[k]]][kv1_word2int[x[k+1]]] for ll in data.keys()]))
             
				A_val = A_val+np.log(np.array([Aall_smooth[ll][kv1_word2int[x[k+1]]][kv1_word2int['</S>']] for ll in data.keys()]))
        		else:
				A_val = A_val+np.log(np.array([Aall_smooth[ll][kv1_word2int[x[0]]][kv1_word2int['</S>']] for ll in data.keys()]))
			#print('probs', A_val)
        		pred_y.append(data.keys()[list(A_val).index(max(A_val))])
        		if pred_y[count] == l: correct+=1
        		val_y.append(l)
			count+=1
		print("abstract #:", count, "y", l, "accuracy", 100*(float(correct)/float(count+1)))
	return pred_y, val_y

def compute_model_stats(pred_y, val_y, data):
	label=[i for i in data.keys()]
	cm=confusion_matrix(val_y, pred_y, labels = label)
	cm = cm.astype(np.float32)
	with open("cm.pickle",'wb') as f:
		pkl.dump(cm, f)

	print("f1 mac",f1_score(val_y, pred_y, average='macro'))
	print("f1 mic",f1_score(val_y, pred_y, average='micro'))
	print("f1 weighted", f1_score(val_y, pred_y, average='weighted'))
	print("prec weighted", precision_score(val_y, pred_y, average='weighted'))
	print("rec weighted", recall_score(val_y, pred_y, average='weighted'))
 
'''        
#####################################################################
data, pubmed_data = get_data("Xtrain_all.json")
kv1,kv2 =  get_chars(pubmed_data)
y_dict = y_embed_code(data)
kv1_word2int, kv1_int2word = alphaint(kv1)
y_dict_word2int, y_dict_int2word = alphaint(y_dict)
print ("size of vocab atleast ten times", len(kv1))
print ("size of out of vocab %", len(kv2))
Nall, Dall = unigram_model(data,kv1,kv1_word2int)
Aall_smooth = laplace_smoothing(Nall, Dall)
Piall = compute_start_state(data,kv1)
pred_y, val_y =  prediction(data, Aall_smooth, kv1,Piall, kv1_word2int)
compute_model_stats(pred_y,val_y,data)

######################################################################
#Test Data
data_test, pubmed_data = get_data("Xdev_all.json")
pred_y_test, val_y_test =  prediction(data_test, Aall_smooth, kv1,Piall, kv1_word2int)
compute_model_stats(pred_y_test, val_y_test, data_test)

with open("pred_y.pickle",'wb') as f:
	pkl.dump(pred_y, f)

with open("val_y.pickle",'wb') as f:
	pkl.dump(val_y, f)

with open("pred_y_test.pickle",'wb') as f:
	pkl.dump(pred_y_test, f)

with open("val_y.pickle",'wb') as f:
        pkl.dump(val_y_test, f)


'''


