import pickle as pkl
import obonet
import json
import numpy as np
import re
import string
from gensim import models, corpora, matutils
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csc_matrix, hstack, vstack

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

def save_json_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

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
        outfile = "../data/all_data.json",'w'
    with open(outfile,'w') as f:
        json.dump(data_dict, f)

def load_pubmed_text_data(filename=None):
    if not filename:
        filename = "../data/pubmed_output.txt"
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
        outfile = "../data/pubmed.json"
        with open(outfile, 'w') as f:
            json.dump(data,f, indent=2)


def create_go_term_vector(filename=None, dump=False):
    if not filename:
        filename = "../data/go.obo"
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
        with open("../data/node_to_index.pkl", 'wb') as f:
            pkl.dump(node_to_index, f)
        with open("../data/index_to_node.pkl", 'wb') as f:
            pkl.dump(index_to_node, f)
    return node_to_index, index_to_node

def get_parent_nodes(node, graph=None):
    if not graph:
        graph = obonet.read_obo("../data/go.obo")
    ans = []
    node_dic = graph._adj[node]
    for every in node_dic:
        for every_key in node_dic[every]:
            if every_key == "is_a":
                ans.append(every)
    return ans


def collect_cleaned_goref_pubmed_data(dump=False):
    with open("../data/GO_REF.pickle", 'rb') as f:
        gorefData = pkl.load(f)
    all_text = []
    all_text_dict = {}
    for i, each in enumerate(gorefData):
        # if i > 19:
        #     break
        text = gorefData[each]
        for line in text:
            line = clean_text(line)
            all_text.append(line)
            all_text_dict[each] = line
    pubmed_data = load_pubmed_text_data()
    for each in pubmed_data:
        line = clean_text(pubmed_data[each]["abstract"])
        all_text.append(line)
        all_text_dict[each] = line
    if dump:
        with open("../data/all_abstract.pickle",'wb') as f:
            pkl.dump(all_text, f)
        with open("../data/all_abstract_withID.pickle", 'wb') as f:
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
        dct = load_pkl_data("../data/dct.pickle")
    corpus = [dct.doc2bow(line) for line in texts]
    return corpus


def get_tfidf_vectors_sparse(corpus, tfidf_model=None):
    if not tfidf_model:
        tfidf_model = load_pkl_data("../data/tfidf_model.pickle")
    vectors = []
    num_terms = len(tfidf_model.idfs)
    for i , each in enumerate(corpus):
        vector = tfidf_model[corpus[i]]
        vectors.append(vector)
    scipy_csc_matrix = matutils.corpus2csc(vectors, num_terms=num_terms)
    return scipy_csc_matrix.T


def create_tfidf_model(documents=None, dump=False):
    if not documents:
        documents = load_pkl_data("../data/all_abstract.pickle")
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]
    dct = corpora.Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]  # convert dataset to BoW format
    model = models.TfidfModel(corpus)  # fit model
    if dump:
        save_pkl_data("../data/dct.pickle", dct)
        save_pkl_data("../data/tfidf_model.pickle", model)
    return model


def create_training_data(data_filename, ontology_filename=None, dump=False):
    node_to_index, index_to_node = create_go_term_vector(ontology_filename)
    json_data = load_json_data(data_filename)
    graph = obonet.read_obo("../data/go.obo")
    abstract_data = load_pkl_data("../data/all_abstract_withID.pickle")
    dct = load_pkl_data("../data/dct.pickle")
    tfidf_model = load_pkl_data("../data/tfidf_model.pickle")
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
        save_pkl_data("../data/data_feature_vector.pickle", features)
        save_pkl_data("../data/data_labels_vector_non_sparse.pickle", labels)






# documents = load_pkl_data("../data/all_abstract.pickle")
# texts = [[word for word in document.lower().split() if word not in stoplist]
#           for document in documents]
# corpus = get_corpus(texts)
# vecs = get_tfidf_vectors_sparse(corpus)
# create_go_term_vector()

# create_training_data("../data/temp_data.json", dump=True)


# json_data = load_json_data("../data/all_data.json")
# abstract_data = load_pkl_data("../data/all_abstract_withID.pickle")
# count = 0
# pcount = 0
# temp_data = []
# pmid_set_np = set()
# pmid_set_p = set()
# for point in json_data:
#     dbref = point["DB_REF"]
#     if "PMID" in dbref:
#         dbref = dbref.split(':')
#         dbref = dbref[1]
#         try:
#             abstract = abstract_data[dbref]
#             pmid_set_p.add(dbref)
#             temp_data.append(point)
#         except:
#             pmid_set_np.add(dbref)
# print("NOT PRESENT ARE %d"%len(pmid_set_np))
# print("PRESENT ARE %d"%len(pmid_set_p))
# save_json_data("../data/temp_data.json", temp_data)