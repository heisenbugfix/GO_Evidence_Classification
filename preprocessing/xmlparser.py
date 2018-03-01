# import nltk
import sys
import gzip
import os
import xml.etree.cElementTree as ET
import pickle as pkl

OUTPUT_FILE = "pubmed_output.txt"
with open('../data/references.pkl','rb') as f:
    ref_id=pkl.load(f)
    ref_pmid=ref_id['PMID']

class Article:
    """
    Class for storing information about each article as extracted by xpath rules
    """
    def __init__(self):
        self.title = ""
        self.mesh_terms = []
        self.mesh_ids = []
        self.mesh_majors = []
        self.abstract = ""
        self.pmid = ""
        self.year = ""

    def add_title(self, elem):
        if elem.text is not None:
            self.title = elem.text

    def add_mesh(self, elem):
        des_name = ""
        des_id = ""
        des_major = ""
        qual_names = []
        qual_ids = []
        qual_majors = []

        overall_major = "N"

        for c in elem.findall(".//DescriptorName"):
            des_name = c.text
            des_id = c.attrib['UI']
            des_major = c.attrib['MajorTopicYN']
        for c in elem.findall(".//QualifierName"):
            qual_names.append(c.text)
            qual_ids.append(c.attrib['UI'])
            qual_majors.append(c.attrib['MajorTopicYN'])

        if des_major == "Y":
            overall_major = "Y"
        if "Y" in qual_majors:
            overall_major = "Y"

        self.mesh_majors.append(overall_major)
        self.mesh_terms.append(elem.text)
        self.mesh_ids.append(elem.attrib["UI"])

    def format_mesh_terms(self):
        mesh_tuples = [(term, id, label) for term, id, label in zip(self.mesh_terms, self.mesh_ids, self.mesh_majors)]
        mesh_tuples = ["|".join(x) for x in mesh_tuples]
        # TODO finish if we need the major topic labels


    def add_pmid(self, elem):
        self.pmid = elem.text

    def add_abstract(self, elem):
        if elem.text is not None:
            self.abstract = elem.text

    def add_year(self, elem):
        self.year = elem.text

    def tab_output(self):
        return "{pmid}\t{year}\t{title}\t{abstract}\t{mesh_terms}\t{mesh_ids}\n".format(
                pmid=self.pmid,
                year=self.year,
                title=self.title.replace("\t"," ").replace("\n"," ").replace("\r"," "),
                abstract=self.abstract.replace("\t"," ").replace("\n"," ").replace("\r"," "),
                mesh_terms="|".join(self.mesh_terms),
                mesh_ids="|".join(self.mesh_ids))

def parse_pubmed(data_dir):
    # Pubmed docs are in tarballs, so we have to open the tarballs
    count,total=0,0
    with open(OUTPUT_FILE, 'w') as output_handle:
        for fn in os.listdir(data_dir):
            if fn.endswith(".xml.gz"):
                print(count,total,fn)
                f = gzip.open(data_dir + os.sep + fn)
                doc = ET.ElementTree(file=f)
                root = doc.getroot()
                art=doc.findall('PubmedArticle')
                total+=len(art)
                for elem in art:
                    for pmid in elem.findall('.//MedlineCitation/PMID'):
                        if pmid.text in ref_pmid:
                            count+=1
                            article = Article()
                            # if elem.tag == "PubmedArticle":
                                # We have an article, now parse it
                            for c in elem.findall('.//Article/ArticleTitle'):
                                # print(c.tag, c.text)
                                article.add_title(c)
                            for c in elem.findall('.//PubDate/Year'):
                                article.add_year(c)
                                # print(c.text)
                            for c in elem.findall('.//MeshHeadingList/MeshHeading/DescriptorName'):
                                # print(c.tag, c.attrib, c.text)
                                article.add_mesh(c)
                            for c in elem.findall('.//PubmedData/ArticleIdList/ArticleId'):
                                # print(c.tag, c.attrib, c.text)
                                if c.attrib['IdType'] == "pubmed":
                                    article.add_pmid(c)
                            for c in elem.findall('.//Article/Abstract/AbstractText'):
                                # print(c.tag, c.attrib, c.text)
                                article.add_abstract(c)
                            elem.clear()
                            output_handle.write(article.tab_output())
                root.clear()

#parse_pubmed(sys.argv[1])



parse_pubmed('/home/twinkle/696oracle/abc')

