import pickle as pkl
import operator
import numpy as np
import random
from numpy.random import choice
from sklearn.metrics import f1_score

with open('../data/new_annotations.pkl','rb') as file:
	rec=pkl.load(file)
	

train_len=int(len(rec)*0.8)
train=rec[:train_len]
test=rec[train_len:]


per_ann={}
per_id={}	
ev_counts={}
#tuples={}
for each in train:
	go_id=each[4]
	ev=each[6]
	journal,ref_id=each[5].split(':',1)
	if ev not in ev_counts:
		ev_counts[ev]=1
	else:
		ev_counts[ev]+=1

	if go_id not in per_ann:
		per_ann[go_id]={}
		if ev not in per_ann[go_id]:
			per_ann[go_id][ev]=1
		else:
			per_ann[go_id][ev]+=1
	else:
		if ev not in per_ann[go_id]:
			per_ann[go_id][ev]=1
		else:
			per_ann[go_id][ev]+=1

	if ref_id not in per_id:
		per_id[ref_id]={}
		if ev not in per_id[ref_id]:
			per_id[ref_id][ev]=1
		else:
			per_id[ref_id][ev]+=1
	else:
		if ev not in per_id[ref_id]:
			per_id[ref_id][ev]=1
		else:
			per_id[ref_id][ev]+=1


#print (per_ann[go_id])
print ('per go_id',len(per_ann))
print ('per ref_id',len(per_id))

mc_ann={}
mc_id={}
for key in per_ann.keys():
	mc_ann[key]=max(per_ann[key].items(),key=operator.itemgetter(1))[0]
#print ('mc go_id',len(mc_ann),mc_ann[key])
for key in per_id.keys():
	mc_id[key]=max(per_id[ref_id].items(),key=operator.itemgetter(1))[0]
#print ('mc ref_id',len(mc_id),mc_id[key])



for key in ev_counts.keys():
	ev_counts[key]=ev_counts[key]/len(train)
evidences=list(ev_counts.keys())
probs=list(ev_counts.values())
ev_nums=np.arange(len(evidences))

y_test=[]
y_pred_ann=[]
y_pred_id=[]
c1=0
c2=0
for each in test:
	go_id=each[4]
	ev=each[6]
	journal,ref_id=each[5].split(':',1)
	y_test.append(ev)
	if go_id in mc_ann:
		y_pred_ann.append(mc_ann[go_id])
		c1+=1
	else:
		y_pred_ann.append(evidences[choice(ev_nums,1,probs)[0]])
		#random.choices(evidences,probs,k=1)[0]

	if ref_id in mc_id:
		y_pred_id.append(mc_id[ref_id])
		c2+=1
	else:
		y_pred_id.append(evidences[choice(ev_nums,1,probs)[0]])
		#random.choices(evidences,probs,k=1)[0]

print ('Test set length',len(y_test))
ann_micro=f1_score(y_test,y_pred_ann,average='micro')
ann_macro=f1_score(y_test,y_pred_ann,average='macro')
ann_w=f1_score(y_test,y_pred_ann,average='weighted')
id_micro=f1_score(y_test,y_pred_id,average='micro')
id_macro=f1_score(y_test,y_pred_id,average='macro')
id_w=f1_score(y_test,y_pred_id,average='weighted')

print ('ANNOTATION BASED F1 scores')
print ("% estimated by existing annotations ",c1/len(y_test))
print ('Micro',ann_micro)
print ('Macro',ann_macro)
print ('Weighted',ann_w)
print ('REFERENCE BASED F1 scores')
print ("% estimated by existing references ",c2/len(y_test))
print ('Micro',id_micro)
print ('Macro',id_macro)
print ('Weighted',id_w)
