import re
import os
import math

class Retrieval:

    def __init__(self):
        postings = self.indexTextFiles_BR('../bbcsport/docs/')
        print('Please enter a query')
        query = input()
        print(self.query_BR(postings, query))

    def readfile(self, path, docid):
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
        return tokens

    def indexTextFiles_BR(self, path):
        N = len(sorted(os.listdir(path)))
        postings = {}
        for docID in range(N):
            s = self.readfile(path, docID)
            tokens = self.tokenize(s)
            for t in tokens:
                postings.setdefault(t, set()).add(docID)
        return postings

    def query_BR(self, postings, query):
        tokens = self.tokenize(query)
        result = None
        for t in tokens:
            result = postings[t] if result is None else result & postings[t]
        return result

    def indexTextFiles_RR(self, path):
        N = len(sorted(os.listdir(path)))
        postings = {}
        for docID in range(N):
            s = self.readfile(path, docID)
            tokens = self.tokenize(s)
            for t in tokens:
                # calculate the tf*idf
                weight = self.calculateTF(t, docID, path) * self.calculateDF(t, N, path)
                #Create an array to hold docid and it's weight
                docAndWeight = [docID, weight]
                postings.setdefault(t, set().add(docAndWeight))
        return postings

    def calculateTF(self, token, docID, path):
        # calculate term frequency
        s = self.readfile(path, docID)
        return s.count(token)

    def calculateDF(self, token, N, path):
        #calculate document frequency
        df = 0
        for docID in range(N):
            s = self.readfile(path, docID)
            if s.__contains__(token):
                df += 1
        idf = math.log(N/df, 10)
        return idf

    def l2normalize(self, postings):
        # find l2norm


    def query_RR(self):



Retrieval()