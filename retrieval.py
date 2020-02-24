import re
import os
import math
import numpy as np
import pandas as pd

class Retrieval:

    documents = []

    def __init__(self):
        self.initialize_documents('../bbcsport/docs/')
        postings = self.index_text_files_rr('../bbcsport/docs/')
        print(postings)
        # print('Please enter a query')
        # query = input()
        # print(self.query_br(postings, query))

    def initialize_documents(self, path):
        N = len(sorted(os.listdir(path)))
        for docID in range(N):
            s = self.read_file(path,docID).lower()
            self.documents.append(s)

    def read_file(self, path, docid):
        files = sorted(os.listdir(path))
        f = open(os.path.join(path, files[docid]), 'r', encoding='latin-1')
        s = f.read()
        f.close()
        return s

    def tokenize(self, string):
        DELIM = '[ \n\t0123456789;:.,/\(\)\"\'-]+'
        # Normalize by removing capitals and removing commas an delimiters
        tokens = re.split(DELIM, string.lower())
        if '' in tokens:
            tokens.remove('')
        #remove duplicates
        return list(dict.fromkeys(tokens))

    def index_text_files_br(self, path):
        N = len(sorted(os.listdir(path)))
        postings = {}
        for docID in range(N):
            s = self.read_file(path, docID)
            tokens = self.tokenize(s)
            for t in tokens:
                postings.setdefault(t, set()).add(docID)
        return postings

    def query_br(self, postings, query):
        tokens = self.tokenize(query)
        result = None
        for t in tokens:
            result = postings[t] if result is None else result & postings[t]
        return result

    def index_text_files_rr(self, path):
        N = len(sorted(os.listdir(path)))
        postings_matrix = pd.DataFrame()
        for docID in range(N):
            s = self.documents[docID]
            tokens = self.tokenize(s)
            row_data = {}
            row_name = ''
            for t in tokens:
                # calculate the tf*idf
                s = s.lower()
                weight = self.calculate_term_freq(t, s) * self.calculate_doc_freq(t, N)
                row_name = docID
                row_data.setdefault(t, weight)
            # create term document matrix
            items = row_data.values()
            df = pd.DataFrame(data=[items], index=[row_name], columns=list(row_data.keys()))
            postings_matrix.append(df)
        return postings_matrix

    def calculate_term_freq(self, token, string):
        # calculate term frequency
        return string.count(token)

    def calculate_doc_freq(self, token, N, ):
        #calculate document frequency
        df = 0
        for doc in self.documents:
            if token in doc:
                df += 1
        idf = math.log(N/df, 10)
        return idf

    def l2normalize(self, docVectors):
        # find l2norm
        l2n = 0
        for v in docVectors:
            l2n += v**2
        l2n = math.sqrt(l2n)
        result = []
        # normalize lengths of vectors
        for v in docVectors:
            result.append(v/l2n)
        return result


    #   def query_RR(self):



Retrieval()