import json
import numpy as np
import io

#p = "/Users/nehachoudhary/Documents/Oracle_Lab/Character_Level/Data/"
#p = "../Data/"

pa ='/home/nchoudhary/train_dev_test/'

labels = ["DB","DB_OID","DB_OBS",
          "Qualifier","GO_ID","DB_REF","EVIDENCE",
          "WITH","Aspect","DB_OBN","DB_OBSYN",
          "DB_OBType","Taxon","Date", "Assigned_By",
          "Annotation_EXT","Gene_PFID"]

evidence_codes = ["EXP", "IDA", "IPI",
                  "IMP", "IGI", "IEP",
                  "HTP", "HDA", "HMP",
                  "HGI", "HEP","ISS",
                  "ISO", "ISA","ISM", 
                  "IGC", "IBA","IBD",
                  "IKR", "IRD","RCA",
                  "TAS", "NAS","IC", 
                  "ND"]

def read_json_input(data):
	with open( data , "r") as of:
		params = json.load(of)
	return params

def get_evidence_codes(input_json):
	evidence_dict = {}
	for line in input_json:
		evidence_dict[line['DB_REF'][5:]] = { 'code' : line['EVIDENCE']}
	return evidence_dict

def load_pubmed_text_data(filename=None):
	if not filename:
		filename = "pubmed_latest.txt"
	data = {}
	with io.open(filename, 'r', encoding='latin1')as f:
		for each in f:
			splitted = each.split('\t',2)
			pubid = splitted[0]
			pubyr = splitted[1]
			abstract = splitted[2]
			data[pubid] = {"date": pubyr, "abstract": abstract}
	return data

def dump_pubmed_json_fromtext(infile=None, outfile=None):
	data = load_pubmed_text_data()
	if not outfile:
		outfile = path + "pubmed.json"
		with open(outfile, 'w') as f:
			json.dump(data,f, indent=2)
	return outfile

def read_json_pubmed(location = None):
        if not location:
                location = "pubmed_latest.json"
        with open( location , "r") as of:
                        data = json.load(of)
        return data

#params = read_json_input("Xtrain_all.json")
#print(params)
#evidence_dict = get_evidence_codes(params)
#pubmed_data = read_json_pubmed()
#pubmed_data = read_json_pubmed()
#xtest = read_json_input_1()
#print(pubmed_data)
#pubmed_json =  dump_pubmed_json_fromtext()








