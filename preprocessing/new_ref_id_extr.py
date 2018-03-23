import pickle as pkl

path='../local/goa_uniprot_all.gaf'
records=[]
ref=[]
with open(path,'r') as annotation:
    for line in annotation:
        if not line.startswith('!'):
            fields=line.split('\t')
            if 'IEA' in fields[6]:
                continue
            records.append(fields)
            ref.append(fields[5])
with open('../data/new_annotations.pkl','wb') as f:
    pkl.dump(records,f)
print('Done writing ',len(records),'records.')

references={}
reflist=[]
golist=[]
for each in ref:
    journal,id=each.split(':',1)
    if journal not in references:
        references[journal]={id}
    else:
        references[journal].add(id)
    if 'PMID' in journal:
        reflist.append(id)
    if 'GO_REF' in journal:
        golist.append(id)

print ('PMID:',len(references['PMID']))
print ('GO_REF:',len(references['GO_REF']))
print ('reflist contains: ',len(reflist))
print ('golist contains',len(golist))

with open('../data/new_references.pkl','wb') as f:
    pkl.dump(references,f)
print('Done writing references')

