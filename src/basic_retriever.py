# Step 0. Load libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, text_fields, keyword_fields, vectorizer_params={}):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.vectorizer = {field: TfidfVectorizer(**vectorizer_params)
                           for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.docs = []

    def fit(self, docs):
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]
            self.text_matrices[field] = (
                self.vectorizer[field].fit_transform(texts)
            )

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10):
        query_vecs = {field: self.vectorizer[field].transform([query])
                      for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        for field, query_vec in query_vecs.items():
            sim = (
                cosine_similarity(query_vec, self.text_matrices[field])
                .flatten()
            )
            boost = boost_dict.get(field, 1)
            scores += sim * boost

        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores *= mask.to_numpy()

        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_docs = [self.docs[i] for i in top_indices if scores[i] > 0]
        return top_docs
