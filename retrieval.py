import re
import os


class BooleanRetrieval:

    def __init__(self):
        postings = self.index('../bbcsport/docs/')
        print('Please enter a query')
        query = input()
        print(self.query(postings, query))

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

    def index(self, path):
        N = len(sorted(os.listdir(path)))
        postings = {}
        for docID in range(N):
            s = self.readfile(path, docID)
            tokens = self.tokenize(s)
            for t in tokens:
                postings.setdefault(t, set()).add(docID)
        return postings

    def query(self, postings, query):
        tokens = self.tokenize(query)
        result = None
        for t in tokens:
            result = postings[t] if result is None else result & postings[t]
        return result

BooleanRetrieval()