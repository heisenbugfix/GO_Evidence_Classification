from __future__ import division
import pickle as pkl
import obonet
import json
import numpy as np
import re
import string
import random
from gensim import models, corpora, matutils
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csc_matrix, hstack, vstack
from Bio.Entrez import efetch

data_directory = "../data/"
labels = ["DB","DB_OID","DB_OBS",
          "Qualifier","GO_ID","DB_REF","EVIDENCE",
          "WITH","Aspect","DB_OBN","DB_OBSYN",
          "DB_OBType","Taxon","Date", "Assigned_By",
          "Annotation_EXT","Gene_PFID"]

evidence_codes = ["EXP", "IDA", "IPI",
                  "IMP", "IGI", "IEP",
                  "HTP", "HDA", "HMP",
                  "HGI", "HEP",
                  "ISS", "ISO", "ISA",
                  "ISM", "IGC", "IBA",
                  "IBD", "IKR", "IRD",
                  "RCA", "TAS", "NAS",
                  "IC", "ND"]

feature_labels = {"GO_ID", "DB_REF", "GO_PARENTS", "Aspect", "Taxon", "DB_OBSYN", "WITH"}
stoplist = set('for a of the and to in'.split())
regex = re.compile('[%s]' % re.escape(string.punctuation))
port_stemmer = PorterStemmer()

def load_pkl_data(filename):
    f = open(filename, 'rb')
    data = pkl.load(f)
    return data

def save_json_data(filename, data, indent=None):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent)

def save_pkl_data(filename, data):
    with open(filename, 'wb') as f:
        pkl.dump(data, f)


def load_json_data(filename):
    f = open(filename)
    data = json.load(f)
    return data

def dump_to_json(filename, outfile=None):
    # dat = load_data("../data/new_annotations.pkl")
    dat = load_pkl_data(filename)
    data_dict = []
    for each in dat:
        curr = {}
        for val, label in zip(each, labels):
            curr[label] = val
        data_dict.append(curr)
    if outfile is None:
        outfile = data_directory+"all_data.json"
    with open(outfile,'w') as f:
        json.dump(data_dict, f)

def load_pubmed_text_data(filename=None):
    if not filename:
        filename = data_directory+"pubmed_output_1.txt"
    data = {}
    with open(filename, 'r', encoding='latin1')as f:
        for each in f:
            splitted = each.split('\t',2)
            pubid = splitted[0]
            pubyr = splitted[1]
            abstract = splitted[2]
            data[pubid] = {"date": pubyr, "abstract": abstract}
    return data

def dump_pubmed_json_fromtext(infile=None, outfile=None):
    data = load_pubmed_text_data(infile)
    if not outfile:
        outfile = data_directory + "pubmed.json"
        with open(outfile, 'w') as f:
            json.dump(data,f, indent=2)


def create_go_term_vector(filename=None, dump=False):
    if not filename:
        filename = data_directory+ "go.obo"
    graph = obonet.read_obo(filename)
    nodes = graph._adj.keys()
    node_to_index = {}
    index_to_node = {}
    for i, node in enumerate(nodes):
        node_to_index[node] = i
        index_to_node[i] = node
    # ohenc = OneHotEncoder(n_values=len(node_to_index))
    # data = ohenc.fit([[1],[3456],[234],[4367]])
    if dump:
        with open(data_directory + "node_to_index.pkl", 'wb') as f:
            pkl.dump(node_to_index, f)
        with open(data_directory + "index_to_node.pkl", 'wb') as f:
            pkl.dump(index_to_node, f)
    return node_to_index, index_to_node

def get_parent_nodes(node, graph=None):
    if not graph:
        graph = obonet.read_obo(data_directory + "go.obo")
    ans = []
    node_dic = graph._adj[node]
    for every in node_dic:
        for every_key in node_dic[every]:
            if every_key == "is_a":
                ans.append(every)
    return ans


def collect_cleaned_goref_pubmed_data(pumed_json_filename, go_ref=False, dump=False):
    all_text = []
    all_text_dict = {}
    if go_ref:
        with open(data_directory + "GO_REF.pickle", 'rb') as f:
            gorefData = pkl.load(f)
        for i, each in enumerate(gorefData):
            text = gorefData[each]
            for line in text:
                line = clean_text(line)
                all_text.append(line)
                all_text_dict[each] = line
    pubmed_data = load_json_data(pumed_json_filename)
    for each in pubmed_data:
        line = clean_text(pubmed_data[each]["abstract"])
        all_text.append(line)
        all_text_dict[each] = line
    if dump:
        with open(data_directory + "all_abstract.pickle",'wb') as f:
            pkl.dump(all_text, f)
        with open(data_directory + "all_abstract_withID.pickle", 'wb') as f:
            pkl.dump(all_text_dict, f)
    return all_text, all_text_dict


def clean_text(line, stemming=False):
    line = line.lower()
    # remove urls (courtesy : stackoverflow)
    line = re.sub(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        '', line)
    new_line = []
    # tokenizing
    tokenized_line = word_tokenize(text=line)
    for token in tokenized_line:
        # removing punctuation and stop words
        if '-' in token and len(token) > 3:
            token = token.split('-')
            new_token = ' '.join(token)
        else:
            new_token = regex.sub(u'', token)
        if not new_token == u'' and not new_token in stopwords.words('english') and new_token not in stoplist:
            if stemming:
                new_token = port_stemmer.stem(new_token)
            new_line.append(new_token)
    return ' '.join(new_line)

def get_evidence_code_dict():
    evid = {}
    for i, each in enumerate(evidence_codes):
        evid[each] = i
    return evid


def get_corpus(texts, dct=None):
    # use a pre-saved dictionary for abstracts
    if not dct:
        dct = load_pkl_data(data_directory + "dct.pickle")
    corpus = [dct.doc2bow(line) for line in texts]
    return corpus


def get_tfidf_vectors_sparse(corpus, tfidf_model=None):
    if not tfidf_model:
        tfidf_model = load_pkl_data(data_directory + "tfidf_model.pickle")
    vectors = []
    num_terms = len(tfidf_model.idfs)
    for i , each in enumerate(corpus):
        vector = tfidf_model[corpus[i]]
        vectors.append(vector)
    scipy_csc_matrix = matutils.corpus2csc(vectors, num_terms=num_terms)
    return scipy_csc_matrix.T


def create_tfidf_model(documents=None, dump=False):
    if not documents:
        documents = load_pkl_data(data_directory + "all_abstract.pickle")
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]
    dct = corpora.Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]  # convert dataset to BoW format
    model = models.TfidfModel(corpus)  # fit model
    if dump:
        save_pkl_data(data_directory + "dct.pickle", dct)
        save_pkl_data(data_directory + "tfidf_model.pickle", model)
    return model


def create_training_data(data_filename, ontology_filename=None, dump=False, feature_filename=None, label_filename=None):
    node_to_index, index_to_node = create_go_term_vector(ontology_filename)
    json_data = load_json_data(data_filename)
    graph = obonet.read_obo(data_directory + "go.obo")
    abstract_data = load_pkl_data(data_directory + "all_abstract_withID.pickle")
    dct = load_pkl_data(data_directory + "dct.pickle")
    tfidf_model = load_pkl_data(data_directory + "tfidf_model.pickle")
    evid_dict = get_evidence_code_dict()
    lab = {"GO_ID", "DB_REF", "GO_PARENTS", "Aspect", "Taxon", "DB_OBSYN", "WITH"}
    Aspect = {"P":0, "F":1, "C":2}
    features = []
    labels = []
    for point in json_data:
        #GO VECTOR
        goterm = point["GO_ID"]
        go_one_hot = np.zeros(len(node_to_index))
        go_one_hot[node_to_index[goterm]] = 1.0
        go_one_hot = csc_matrix(go_one_hot)
        # ABSTRACT VECTOR
        dbref = point["DB_REF"]
        if "PMID" in dbref:
            dbref = dbref.split(':')
            dbref = dbref[1]
        abstract = abstract_data[dbref]
        abstract = [[word for word in abstract.lower().split() if word not in stoplist]]
        corpus = get_corpus(abstract, dct)
        abstract_vec = get_tfidf_vectors_sparse(corpus, tfidf_model)
        feature = hstack([go_one_hot, abstract_vec])
        #GO_PARENTS
        parents = get_parent_nodes(goterm, graph)
        parent_vec = np.zeros(len(node_to_index))
        for each in parents:
            parid =node_to_index[each]
            parent_vec[parid] = 1.0
        parent_vec = csc_matrix(parent_vec)
        feature = hstack([feature, parent_vec])
        #ASPECT:
        aspect_one_hot = np.zeros(len(Aspect))
        aspect_one_hot[Aspect[point["Aspect"]]] = 1.0
        aspect_one_hot = csc_matrix(aspect_one_hot)
        feature = hstack([feature, aspect_one_hot])
        #Label
        evd_c = point["EVIDENCE"]
        # evd_one_hot = np.zeros(len(evidence_codes))
        # evd_one_hot[evid_dict[evd_c]] = 1.0
        # evd_one_hot = csc_matrix(evd_one_hot)
        #stack all data
        features.append(feature)
        labels.append(evid_dict[evd_c])
    features = vstack(features)
    labels = np.asarray(labels)
    if dump:
        if feature_filename is None:
            feature_filename = "data_feature_vector.pickle"
        save_pkl_data(data_directory + feature_filename, features)
        if label_filename is None:
            label_filename = "data_labels_vector_non_sparse.pickle"
        save_pkl_data(data_directory + label_filename, labels)


def get_abstracts_from_pmid(pmid):
    handle = efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
    return (handle.read())


def create_raw_data(filename=None):
    data_unique_annotations_pmids = []
    data_complex_pmids = []
    if filename is None:
        filename = "../data/all_data.json"
    all_data = load_json_data(filename)
    available_pmid = load_json_data("../data/pubmed.json")
    u_pmid = load_pkl_data("../data/unique_PMID.pickle")
    pmid_not_present = set()
    is_presence_of_extra_pmids = False
    for datapoint in all_data:
        if "PMID" in datapoint["DB_REF"]:
            pmid = datapoint["DB_REF"]
            pmid = pmid.split(':')
            pmid = pmid[1]
            if pmid in available_pmid:
                data_complex_pmids.append(datapoint)
                if datapoint["DB_REF"] in u_pmid:
                    data_unique_annotations_pmids.append(datapoint)
            else:
                try:
                    is_presence_of_extra_pmids = True
                    abstract = get_abstracts_from_pmid(pmid)
                    if abstract is not None and len(abstract) > 0:
                        available_pmid[pmid] = {}
                        available_pmid[pmid]["abstract"] = abstract
                        data_complex_pmids.append(datapoint)
                        if datapoint["DB_REF"] in u_pmid:
                            data_unique_annotations_pmids.append(datapoint)
                    else:
                        print("IN TRY!! COULD NOT FIND PMID GLOBALLY")
                        pmid_not_present.add(pmid)
                except:
                    print("COULD NOT FIND PMID GLOBALLY")
                    pmid_not_present.add(pmid)

    save_json_data("../data/pmids_not_present.json", list(pmid_not_present))
    save_json_data("../data/pubmed_latest.json", available_pmid, indent=1)
    print("NUMBER OF PMIDS NOT AVAILABLE ARE: %d"%len(pmid_not_present))
    print("TOTAL DATA POINTS FOR COMPLEX DATA IS %d"%len(data_complex_pmids))
    print("TOTAL DATA POINTS FOR UNIQUE ANNOTATION PMID IS %d"%len(data_unique_annotations_pmids))
    save_json_data("../data/data_with_all_pmids.json", data_complex_pmids)
    save_json_data("../data/data_with_uniqueAnnotation_pmid.json", data_unique_annotations_pmids)

def create_unique_abstract_dict():
    unique = set()
    d = load_pkl_data("../data/dbref_annotations.pickle")
    count = 0
    for each in d:
        if "PMID" in each:
            if d[each] == 1:
                count += 1
                unique.add(each)
    save_pkl_data("../data/unique_PMID.pickle", unique)
    print("Number of PMIDS with unique annotation are: %d"%count)

def create_train_dev_test():
    unique_data = load_json_data("../data/data_with_uniqueAnnotation_pmid.json")
    all_data = load_json_data("../data/data_with_all_pmids.json")
    random.shuffle(unique_data)
    random.shuffle(all_data)
    udata_len = len(unique_data)
    all_data_len = len(all_data)
    #create train/dev/test for unique ones
    Xtrain_unique, Xdev_unique, Xtest_unique = split_data_helper(udata_len, unique_data)
    #create train/dev/test for all
    Xtrain_all, Xdev_all, Xtest_all = split_data_helper(all_data_len, all_data)
    save_json_data("../data/final_data/Xtrain_unique.json", Xtrain_unique)
    save_json_data("../data/final_data/Xdev_unique.json", Xdev_unique)
    save_json_data("../data/final_data/Xtest_unique.json", Xtest_unique)

    save_json_data("../data/final_data/Xtrain_all.json", Xtrain_all)
    save_json_data("../data/final_data/Xdev_all.json", Xdev_all)
    save_json_data("../data/final_data/Xtest_all.json", Xtest_all)

def split_data_helper(data_len, data):
    train_len = int(0.7 * data_len)
    rem_len = data_len - train_len
    dev_len = int(rem_len * 2 / 3)
    Xtrain = data[: train_len]
    Xdev = data[train_len: train_len+dev_len]
    Xtest = data[train_len+dev_len : ]
    return Xtrain, Xdev, Xtest



# collect_cleaned_goref_pubmed_data(data_directory+"final_data/train_dev_test/pubmed_latest.json", dump=True)
# create_tfidf_model(dump=True)
# create_training_data("../data/final_data/train_dev_test/unique/Xdev_unique.json",dump=True,feature_filename="Xdev_unique.pickle",label_filename="Ydev_unique.pickle")
# create_training_data("../data/final_data/train_dev_test/all/Xdev_all.json",dump=True,feature_filename="Xdev.pickle",label_filename="Ydev.pickle")

# create_train_dev_test()
# create_raw_data()
# create_unique_abstract_dict()

# dump_pubmed_json_fromtext("../data/pubmed_output_1.txt")
# create_tfidf_model()
# collect_cleaned_goref_pubmed_data(dump=True)
# print("OK")
# documents = load_pkl_data("../data/all_abstract.pickle")
# texts = [[word for word in document.lower().split() if word not in stoplist]
#           for document in documents]
# corpus = get_corpus(texts)
# vecs = get_tfidf_vectors_sparse(corpus)
# create_go_term_vector()

# create_training_data("../data/temp_data.json", dump=True)
# data = load_json_data(data_directory+"temp_data.json")
# print (len(data))

# json_data = load_json_data("../data/all_data.json")
# abstract_data = load_pkl_data("../data/all_abstract_withID_1.pickle")
# dbref_dict_evidence_code = {}
# dbref_annotations = {}
# gene_id_annotations = {}
# gene_id_evidence = {}
#
# count = 0
# pcount = 0
# pmid_count = 0
# go_ref_count = 0
# other_count = 0
# temp_data = []
# pmid_set_np = set()
# pmid_set_p = set()
#
# total_count = 0
# for point in json_data:
#     total_count+=1
#     dbref = point["DB_REF"]
#     evidence = point["EVIDENCE"]
#     geneID = point["GO_ID"]
#     #####################
#     if dbref in dbref_dict_evidence_code:
#         dbref_dict_evidence_code[dbref].add(evidence)
#     else:
#         dbref_dict_evidence_code[dbref] = set(evidence)
#     if dbref in dbref_annotations:
#         dbref_annotations[dbref]+=1
#     else:
#         dbref_annotations[dbref] = 1
#
#     if geneID in gene_id_evidence:
#         gene_id_evidence[geneID].add(evidence)
#     else:
#         gene_id_evidence[geneID] = set(evidence)
#     if geneID in gene_id_annotations:
#         gene_id_annotations[geneID]+=1
#     else:
#         gene_id_annotations[geneID] = 1
#
#     #####################
#     if "PMID" in dbref:
#         dbref = dbref.split(':')
#         dbref = dbref[1]
#         pmid_count+=1
#         try:
#             abstract = abstract_data[dbref]
#             pmid_set_p.add(dbref)
#             temp_data.append(point)
#         except:
#             pmid_set_np.add(dbref)
#     elif "GO_REF" in dbref:
#         go_ref_count+=1
#     else:
#         other_count+=1
#
# print("PMID COUNT IS %d which is %f of total count"%(pmid_count, pmid_count/total_count))
# print("GO_REF COUNT IS %d which is %f of total count"%(go_ref_count, go_ref_count/total_count))
# print("OTHERS COUNT IS %d which is %f of total count"%(other_count, other_count/total_count))
# print("TOTAL COUNT IS %d"%total_count)
#
# count_dbref_annotations = 0
# count_geneID_annotations = 0
# count_dbref_evidence_code = 0
# count_geneID_evidence_code = 0
#
# for each in dbref_annotations:
#     count_dbref_annotations += dbref_annotations[each]
#
# for each in dbref_dict_evidence_code:
#     count_dbref_evidence_code += len(dbref_dict_evidence_code[each])
#
# for each in gene_id_annotations:
#     count_geneID_annotations += gene_id_annotations[each]
#
# for each in gene_id_evidence:
#     count_geneID_evidence_code += len(gene_id_evidence[each])
#
# print("Average number of annotations per GENE ID is %f"%(count_geneID_annotations/len(gene_id_annotations)))
# print("Average number of evidence codes per GENE ID is %f"%(count_geneID_evidence_code/len(gene_id_evidence)))
#
# print("Average number of annotations per ABSTRACT ID is %f"%(count_dbref_annotations/len(dbref_annotations)))
# print("Average number of evidence codes per ABSTRACT ID is %f"%(count_dbref_evidence_code/len(dbref_dict_evidence_code)))
#
# print("MAX NUMBER OF ANNOTATIONS FOR AN ABSTRACT IS %d"%(max(dbref_annotations.values())))
# print("MAX NUMBER OF ANNOTATIONS FOR A GENE ID IS %d"%(max(gene_id_annotations.values())))
# save_pkl_data("../data/dbref_annotations.pickle", dbref_annotations)
# save_pkl_data("../data/gene_id_annotations.pickle", gene_id_annotations)

# d = load_pkl_data("../data/dbref_annotations.pickle")
# print("OK")

# print("NOT PRESENT ARE %d"%len(pmid_set_np))
# print("PRESENT ARE %d"%len(pmid_set_p))
# with open("not_present.txt", 'w') as f:
#     for each in pmid_set_np:
#         f.write(str(each)+"\n")
# save_json_data("../data/temp_data.json", temp_data)


# json_data = load_json_data("../data/temp_data.json")
# i = 0
# for each in json_data:
#     i+=1
# print(i)


