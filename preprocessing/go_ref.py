from collections import defaultdict
from bs4 import BeautifulSoup
import requests
import urllib, json

#function: yields groups of <p> nodes separated by <div> siblings
def chunk(parent):    
    chunk = []
    for element in parent.find_all():
        if element.name == 'p':
            chunk.append(element)
        elif element.name == 'div':
            yield chunk
            chunk = []
    if chunk:
        yield chunk

page_link ='http://www.geneontology.org/cgi-bin/references.cgi'
page_response = requests.get(page_link, timeout=5)
page_content = BeautifulSoup(page_response.content, "html.parser")
#print(page_content )

name = []
for element in page_content.find_all():
	if element.find('h2') != None:
		name_box =  str(element.find('h2'))
		for ch in ['<h2>','</h2>']:
			if ch in name_box:
				name_box=name_box.replace(ch,'')	
		name.append(name_box)
#print (name)
goref_kv = {}

for n in name:
	for paras in chunk(page_content.find(class_='block', id = n )):
		val=('\n'.join(p.get_text() for p in paras))
		#val=[p for p in paras]
		goref_kv[n]=[val]   

#print(goref_kv['GO_REF:0000003'])




