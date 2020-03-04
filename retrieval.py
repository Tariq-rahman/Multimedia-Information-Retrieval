import re
import os
import math
import numpy as np
import operator
from collections import OrderedDict


class Retrieval:
    # list of all documents
    documents = []
    # dict containing doc frequency scores for all terms
    document_frequency = {}
    PATH_TO_DOCS = "bbcsport/docs/"
    postings = {}

    def __init__(self):
        self.initialize_documents(self.PATH_TO_DOCS)
        self.initialize_doc_freq(len(self.documents))
        self.postings = self.index_text_files_rr(self.PATH_TO_DOCS)
        # Boolean retrieval code below
        # self.index_text_files_br(self.PATH_TO_DOCS)

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
        return s.lower()

    @staticmethod
    def tokenize(string):
        """
        Tokenizes the input string by normalising and removing duplicates
        :param string:
        :return tokens: a list of tokens
        """
        DELIM = '[ \n\t0123456789;:.,/\(\)\"\'-]+'
        # Normalize by removing capitals and removing commas an delimiters
        tokens = re.split(DELIM, string.lower())
        if '' in tokens:
            tokens.remove('')
        # Remove duplicates and return
        return list(dict.fromkeys(tokens))

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
        if token in self.document_frequency:
            return self.document_frequency[token]
        else:
            # calculate document frequency
            df = 0
            for doc in self.documents:
                if token in doc:
                    df += 1
            idf = math.log(N / df, 10)
            # Store idf score to speed up process for high frequency terms
            self.document_frequency.setdefault(token, idf)
        return idf

    @staticmethod
    def calculate_vector_norm(vector):
        n = 0
        for key, value in vector.items():
            n += value ** 2
        norm = math.sqrt(n)
        return norm

    def l2normalize(self, docVectors):
        """
        Uses the l2norm to normalize the given vectors by dividing each vector by the l2Norm
        :param docVectors: Document vector to normalize
        :return: the normalized vector
        """
        # find l2norm
        norm = self.calculate_vector_norm(docVectors)
        # normalize lengths of vectors
        for key, value in docVectors.items():
            docVectors[key] = value / norm
        return docVectors

    def cosine(self, a, b):
        """
        Calculates the cosine of the angle between two given vectors, a and b
        :param a: Vector a
        :param b: Vector b
        """
        # calculate dot product
        dot_product = np.dot(list(a.values()), list(b.values()))
        # calculate magnitude of a and b
        a_magnitude = self.calculate_vector_norm(a)
        b_magnitude = self.calculate_vector_norm(b)
        return dot_product / (a_magnitude * b_magnitude)

    def initialize_documents(self, path):
        """
        Reads all the documents and stores them in a list to reduce read operations
        and increase process speed
        :param path: The path to the document
        """
        N = len(sorted(os.listdir(path)))
        for docID in range(N):
            s = self.read_file(path, docID)
            self.documents.append(s)

    def initialize_doc_freq(self, N):
        for docID in range(N):
            s = self.documents[docID]
            tokens = self.tokenize(s)
            for t in tokens:
                self.calculate_doc_freq(t, N)

    def index_text_files_rr(self, path):
        """
        Indexes the all the documents and terms into a postings matrix for ranked retrieval
        :param path: the path to the documents
        :return postings_matrix: a term document matrix with contents being the tfidf score
        """
        N = len(sorted(os.listdir(path)))
        postings = {}
        for docID in range(N):
            s = self.documents[docID]
            tokens = self.tokenize(s)
            # Create a template using the keys of the document_frequency dict
            # This will ensure that all doc vectors have same dimensions
            document_data = dict.fromkeys(self.document_frequency, 0)
            for t in tokens:
                # calculate the relevancy score using tf*idf
                tf_idf = self.calculate_term_freq(t, s) * self.document_frequency[t]
                # insert tf_idf score to corresponding terms in dict
                document_data[t] = tf_idf
                # Normalize the document scores
            normalized_data = self.l2normalize(document_data)
            postings.setdefault(docID, normalized_data)
        return postings

    def query_rr(self, query, max_results=10):
        """
        Query function for ranked retrieval
        :param query:
        :param postings:
        :param max_results:
        :return:
        """
        tokens = self.tokenize(query)
        # Create dict with zeros
        tf_idf = dict.fromkeys(self.document_frequency, 0)
        for t in tokens:
            # insert idf scores for tokens, tf scores not needed as it will be 1
            tf_idf[t] = self.document_frequency[t]
        cosine_scores = {}
        for key, value in self.postings.items():
            cosine_scores.setdefault(key, self.cosine(tf_idf, value))
        # Sort in descending order to get top results
        result = OrderedDict(sorted(cosine_scores.items(), key=operator.itemgetter(1), reverse=True)[:max_results])
        return result.keys()

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

    def query_br(self, query, postings):
        """
        Query function for boolean retrieval
        Uses the postings list to get all sets of docs that contain the query term
        then merges the doc lists
        :param postings:
        :param query:
        :return:
        """
        tokens = self.tokenize(query)
        result = None
        for t in tokens:
            result = postings[t] if result is None else result & postings[t]
        return result


retrieval = Retrieval()
print(retrieval.query_rr('england mccall united'))
