import re
import os
import math
import numpy as np
import pandas as pd
import time

class Retrieval:
    # list of all documents
    documents = []
    # dict containing doc frequency scores for all terms
    term_rarity = {}

    def __init__(self):
        self.initialize_documents('../bbcsport/docs/')
        start = time.time()
        postings = self.index_text_files_rr('../bbcsport/docs/')
        end = time.time()
        print(postings)
        print("time taken: " + end-start)
        # print('Please enter a query')
        # query = input()
        # print(self.query_br(postings, query))

    def initialize_documents(self, path):
        """
        Reads all the documents and stores them in a list to reduce read operations
        and increase process speed
        :param path: The path to the document
        """
        # Store documents in array to reduce number of read commands
        N = len(sorted(os.listdir(path)))
        for docID in range(N):
            s = self.read_file(path,docID).lower()
            self.documents.append(s)

    @staticmethod
    def read_file(path, docid):
        """
        Returns the requested document in a string format
        :param path: The path to the document
        :param docid: ID of the document to read
        :return: string of the document
        """
        files = sorted(os.listdir(path))
        f = open(os.path.join(path, files[docid]), 'r', encoding='latin-1')
        s = f.read()
        f.close()
        return s

    @staticmethod
    def tokenize(string):
        """
        Tokenizes the input string by normalising and removing duplicates
        :param string:
        :return:
        """
        DELIM = '[ \n\t0123456789;:.,/\(\)\"\'-]+'
        # Normalize by removing capitals and removing commas an delimiters
        tokens = re.split(DELIM, string.lower())
        if '' in tokens:
            tokens.remove('')
        # Remove duplicates and return
        return list(dict.fromkeys(tokens))

    def index_text_files_br(self, path):
        """
        Indexes all the terms and documents into a postings dictionary for boolean retrieval
        :param path: The path to the documents
        :return postings: term to document dictionary e.g. {term: {0,2,4}, term2:{9,7,4}}
        """
        # Index function for boolean retrieval
        N = len(sorted(os.listdir(path)))
        postings = {}
        for docID in range(N):
            s = self.read_file(path, docID)
            tokens = self.tokenize(s)
            for t in tokens:
                postings.setdefault(t, set()).add(docID)
        return postings

    def query_br(self, postings, query):
        """

        :param postings:
        :param query:
        :return:
        """
        # Query function for boolean retrieval
        tokens = self.tokenize(query)
        result = None
        for t in tokens:
            result = postings[t] if result is None else result & postings[t]
        return result

    def index_text_files_rr(self, path):
        """
        Indexes the all the documents and terms into a postings matrix for ranked retrieval
        :param path: the path to the documents
        :return postings_matrix: a term document matrix with contents being the tfidf score
        """
        N = len(sorted(os.listdir(path)))
        # Create empty dataframe
        postings_matrix = pd.DataFrame()
        for docID in range(N):
            s = self.documents[docID]
            tokens = self.tokenize(s)
            row_data = {}
            col_name = ''
            for t in tokens:
                # calculate the relevancy score using tf*idf
                score = self.calculate_term_freq(t, s.lower()) * self.calculate_doc_freq(t, N)
                col_name = docID
                # Create a row of doc->score data to be inserted in matrix
                row_data.setdefault(t, score)
            # Normalize the document scores
            items = self.l2normalize(row_data.values())
            df = pd.DataFrame(data=[items], index=[col_name], columns=list(row_data.keys()))
            # append to term document matrix
            postings_matrix = postings_matrix.append(df, ignore_index=True)
        # Transpose matrix to make documents as columns and terms as rows
        return postings_matrix.transpose()

    @staticmethod
    def calculate_term_freq(token, doc):
        """
        Calculates the number of times the token appears in the document string
        :param token: The token
        :param doc: The document
        :return: (int) term frequency
        """
        return doc.count(token)

    def calculate_doc_freq(self, token, N, ):
        """
        Calculates the document frequency for a given term (idf)
        The rarer the term, the higher the score
        Since there are many re-occuring terms, they are stored in a dict to reduce processing time
        :param token: The term
        :param N: Total number of documents
        :return idf: The document frequency score
        """
        # Check if token idf has already been calculated before
        if token in self.term_rarity:
            return self.term_rarity[token]
        else:
            # calculate document frequency
            df = 0
            for doc in self.documents:
                if token in doc:
                    df += 1
            idf = math.log(N/df, 10)
            # Store idf score to speed up process for high frequency terms
            self.term_rarity.setdefault(token,idf)
        return idf

    @staticmethod
    def l2normalize(docVectors):
        """
        Uses the l2norm to normalize the given vectors by dividing each vector by the l2Norm
        :param docVectors: Document vector to normalize
        :return: the normalized vector
        """
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