"""
Simple indexer and search engine built on an inverted-index and the BM25 ranking algorithm.
"""
import os
from collections import defaultdict, Counter
import pickle
import math
import operator
import code

from tqdm import tqdm
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset


class Indexer:
    dbfile = "./ir.idx"  # file to save/load the index

    def __init__(self):
        # TODO. You will need to create necessary and appropriate 
        #     data structures for the following elements. The following
        #     are just example placeholders and you can change them as you
        #     see fit.
        # self.tok2idx = None                       # map (token to id)
        # self.idx2tok = None                       # map (id to token)
        # self.postings_lists = None                # postings for each word
        # self.docs = []                            # encoded document list
        # self.raw_ds = None                        # raw documents for result presentation
        # self.corpus_stats = { 'avgdl': 0 }        # any corpus-level statistics
        # self.stopwords = stopwords.words('english')

        if os.path.exists(dbfile):
            # TODO. If these exists a saved corpus index file, load it.
            # (You may use pickle to save and load a python object.)
            pass
        else:
            # TODO. Load CNN/DailyMail dataset, preprocess and create postings lists.
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']
            self.clean_text(self.raw_ds)
            self.create_postings_lists()

    def clean_text(self, lst_text, query=False):
        # TODO. this function will run in two modes: indexing and query mode.
        # TODO. run simple whitespace-based tokenizer (e.g., RegexpTokenizer)
        # TODO. run lemmatizer (e.g., WordNetLemmatizer)
        # TODO. read documents one by one and process
        for l in lst_text:
            # TODO. complete this part
            pass

    # def create_postings_lists(self):
        # TODO. This creates postings lists of your corpus
        # TODO. While indexing compute avgdl and document frequencies of your vocabulary
        # TODO. Save it, so you don't need to do this again in the next runs.
        # Save


class SearchAgent:
    k1 = 1.5                # BM25 parameter k1 for tf saturation
    b = 0.75                # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        self.i = indexer

    def query(self, q_str):
        toks = self.i.clean_text([q_str], query=True)
        doc_scores = defaultdict(int)

        for tok in toks:
            if tok in self.i.tok2idx:
                idx = self.i.tok2idx[tok]
                for docid, tf in self.i.postings_lists[idx]:
                    doc_len = len(self.i.docs[docid])
                    idf = math.log((self.i.corpus_stats['N'] - len(self.i.postings_lists[idx]) + 0.5) / len((self.i.postings_lists[idx]) + 0.5) + 1)
                    doc_scores[docid] += idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.i.corpus_stats['avgdl']))))

        results = sorted(doc_scores.items(), key=operator.itemgetter(1), reverse=True)
        if len(results) == 0:
            return None
        else:
            self.display_results(results)


    def display_results(self, results): 
        for docid, score in results[:5]:  # print top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid])



if __name__ == "__main__":
    i = Indexer()           # instantiate an indexer
    q = SearchAgent(i)      # document retriever
    code.interact(local=dict(globals(), **locals())) # interactive shell