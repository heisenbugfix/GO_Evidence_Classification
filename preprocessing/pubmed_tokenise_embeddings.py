import re
import json
import pickle as pkl
# import preprocess
from genia_tokenizer import *
from gensim.models.keyedvectors import KeyedVectors

# punct={}

def num_tokens(text):
	new_text=[]
	for each in text:
		# print(each)
		# if each==" ":
			# continue
		if re.search('[0-9]',each) is not None:
			if re.search('[a-zA-Z]',each) is None:
				new_text.append('<number>')
		else:

			each=each.lower()
			if each.endswith('.') and re.search('[a-zA-Z]',each) is not None:
				# print(each)
				new_text.append(each[:-1])
				new_text.append('.')
			elif each.endswith('?'):
				new_text.append(each[:-1])
				new_text.append('?')
			else:
				new_text.append(each.lower())	

	final_sent=[]
	if len(new_text)>1:

		#split to sentences based on . or ?
		sent1=[]
		new_text=(' '.join(new_text)).split(' . ')
		# for every . separated sentence
		for each in new_text[:-1]:
			# append . to end of sentence
			each=each.split(' ')
			each.append('.')
			sent1.append(each)
		sent1.append(new_text[-1].split(' '))


		# print(sent1)
		
		sent2=[]
		#for every sentence
		for each in sent1:
			# split by ? into further sentences within
			new_text=(' '.join(each)).split(' ? ')
			# print (new_text)
			
			smtn=[]
			#for each sentence within
			for every in new_text[:-1]:
				every=every.split(' ')
				every.append('?')
				smtn.append(every)
			if len(smtn)>0:
				final_sent.extend(smtn)
			final_sent.append(new_text[-1].split(' '))


		# final_nopunct=[]
		# for i in range(len(final)):
		# 	for j in range(len(final[i])):
		# 		if re.search('[0-9]',final[i][j]) is None:
		# 			if re.search('[a-zA-Z]',final[i][j]) is None:
		# 				if final[i][j] not in punct.keys():
		# 					l=len(punct)
		# 					punct[final[i][j]]='<token'+str(l)+'>'
		# 				key=final[i][j]
		# 				final[i][j]=punct[final[i][j]]

		# 				# print(final[i][j],key)

		# print (final_sent)
	return final_sent




def pubmed_smtn(filename):
	f=open(filename,encoding='latin1')
	data=f.read()
	data=data.split('\n')
	articles={}
	c=1
	for each in data:
		if len(each)<=0:
			continue
		# pmid,year,title,abstract,meshterms,meshid=re.split(r'\s{2,}',each)
		# print(data)
		if c%100==0:
			print(c)
		c+=1
		# parts=re.split(r'\s{2,}',each)
		parts=each.split('\t')
		# print(len(parts),parts)
		pmid=parts[0]
		year=parts[1]
		if len(year)==0:
			year=0
		# title=parts[2]
		# title=preprocess.clean_text(parts[2])
		title=num_tokens(tokenize(parts[2]))
		
		# abstract=parts[3]
		# abstract=preprocess.clean_text(parts[3])
		abstract=num_tokens(tokenize(parts[3]))

		meshterms=tokenize(' '.join(parts[4].split('|')))

		meshid=parts[5].split('|')

		if pmid not in articles:
			articles[pmid]=[]
		# print(year)
		articles[pmid]=[int(year),title,abstract,meshterms,meshid]
	
	# print(articles)
	# print(len(data),len(list(articles.keys())))

	with open('../data/genia_parsed_pubmed_sent.pkl','wb') as f:
		pkl.dump(articles,f)




def create_embeddings():
	with open('../data/genia_parsed_pubmed_sent.pkl','rb') as f:
		articles=pkl.load(f)	

	model = KeyedVectors.load_word2vec_format('../local/PubMed-w2v.bin', binary=True)

	records={}
	count_all=0
	count_not_present=0
	present=set()
	not_present=set()
	itr=0
	for key in articles.keys():
		if key not in records:
			records[key]=[]
		itr+=1
		if itr%100==0:
			print(itr)

		tam=[]
		for each in [articles[key][1],articles[key][2]]:
			embedding=[]
			for sent in each:
				for word in sent:
					count_all+=1
					try:
						embedding.append(model[word])
						present.add(word)
					except:
						count_not_present+=1
						# print(word)
						not_present.add(word)
			tam.append(embedding)

		records[key]=[articles[key][0],tam[0],tam[1],articles[key][3],articles[key][4]]

	with open('../data/genia_bionlp_embeddings_2.pkl','wb') as f:
		pkl.dump(records,f)
	print("count_all= ",count_all)
	print("count_not_present= ",count_not_present)
	print("set count_present= ",len(present))
	print("set count_not_present= ",len(not_present))



	np={'np':list(not_present)}
	with open('../data/genia_bionlp_embeddings_2_notpresent.json','w') as f:
		json.dump(np,f)




# pubmed_smtn('../data/pubmed_output.txt')
create_embeddings()

