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
import nltk

class Indexer:
    nltk.download('averaged_perceptron_tagger_eng')
    dbfile = "./ir.idx"  # file to save/load the index

    def __init__(self):
        # TODO. You will need to create necessary and appropriate 
        #     data structures for the following elements. The following
        #     are just example placeholders and you can change them as you
        #     see fit.
         self.tok2idx = {}                       # map (token to id)
         self.idx2tok = []                       # map (id to token)
         self.postings_lists = defaultdict(dict)                # postings for each word
         self.docs = []                            # encoded document list
         self.raw_ds = None                        # raw documents for result presentation
         self.corpus_stats = { 'N': 0, 'avgdl': 0 }        # any corpus-level statistics
         self.stopwords = stopwords.words('english')
         self.lemmatizer = WordNetLemmatizer()
         self.tokenizer = RegexpTokenizer(r'\w+')

         if os.path.exists(self.dbfile):
            # TODO. If these exists a saved corpus index file, load it.
            # (You may use pickle to save and load a python object.)
           with open(self.dbfile, 'rb') as f:
                self.tok2idx,self.idx2tok,self.postings_lists,self.docs, self.raw_ds = pickle.load(f)
           self.corpus_stats['N'] = len(self.docs)
           self.corpus_stats['avgdl'] = sum([len(doc) for doc in self.docs]) / len(self.docs)
           
            
         else:
            # TODO. Load CNN/DailyMail dataset, preprocess and create postings lists.
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']

            self.clean_text(indexing = True)

            print(f"Total documents processed: {len(self.docs)}")
            if self.docs:
                print(f"Sample document: {self.docs[0]}")


            self.create_postings_lists()
            with open(self.dbfile, 'wb') as f:
                pickle.dump((self.tok2idx, self.idx2tok, self.postings_lists, self.docs, self.raw_ds), f)


    def clean_text(self, query=None, indexing = False):
        lst_text = query if query else self.raw_ds  # Handle both cases
        lengths = []
        processed = []

        print("Cleaning text...")

        print(f"Processing {len(lst_text)} documents...")  # Debugging line

        for l in tqdm(lst_text):
            tokenized = self.tokenizer.tokenize(l.lower())
            tagged = pos_tag(tokenized)
            lemmatized = [self.lemmatizer.lemmatize(word) for word, tag in tagged if word.lower() not in self.stopwords]
            

            if lemmatized:
                processed.append(lemmatized)
                if indexing:  # FIXED: Ensure we append during indexing
                    lengths.append(len(tokenized))
                    self.docs.append(lemmatized)
                    print(f"Added documents of length {len(lemmatized)}")
            else:
                print(f"Skipping empty document after lemmatization: {l}")  # Debugging line
        if indexing:
            self.corpus_stats['N'] = len(self.docs)
            self.corpus_stats['avgdl'] = sum(lengths) / len(lengths) if lengths else 0
            print(f"Corpus stats: {self.corpus_stats}")

        return processed if not indexing else None  # Only return tokens for queries

            

    def create_postings_lists(self):
        # TODO. This creates postings lists of your corpus
        # TODO. While indexing compute avgdl and document frequencies of your vocabulary
        # TODO. Save it, so you don't need to do this again in the next runs.
        # Save
        for doc_id, document in enumerate(self.docs):
            tf = Counter(document)
            for w, tf in tf.items():
                if w not in self.tok2idx:
                    self.tok2idx[w] = len(self.idx2tok)
                    self.idx2tok.append(w)
                word_index = self.tok2idx[w]
                self.postings_lists[word_index][doc_id] = tf
        
        print(f"Number of words in index: {len(self.tok2idx)}")

        

class SearchAgent:
    k1 = 1.5             # BM25 parameter k1 for tf saturation
    b = 0.75                # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        self.i = indexer

    def query(self, q_str):
        toks = self.i.clean_text([q_str])

        print(f"Query tokens: {toks}")  # Debugging line

        if not toks:
            print("No tokens found in the query.")
            return None
        
        toks = [word for sublist in toks for word in sublist]
        doc_scores = defaultdict(float)

        print(f"AvgDL: {self.i.corpus_stats['avgdl']}")

        for tok in toks:
            if tok not in self.i.tok2idx:
                continue
            else:
                word_index = self.i.tok2idx[tok]
                if word_index in self.i.postings_lists:
                    for docid, tf in self.i.postings_lists[word_index].items():
                        doc_len = len(self.i.docs[docid])
                        idf = math.log((self.i.corpus_stats['N'] - len(self.i.postings_lists[word_index]) + 0.5) / (len(self.i.postings_lists[word_index]) + 0.5) + 1)
                       
                        doc_scores[docid] += idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.i.corpus_stats['avgdl']))))

        print("Sorting results...")  # Debugging line
        results = sorted(doc_scores.items(), key=operator.itemgetter(1), reverse=True)
        print("displaying results...")  # Debugging line
        self.display_results(results)


    def display_results(self, results): 
        if not results:
            print("No results found.")
            return
        for docid, score in results[:5]:  # print top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid][:500] + '...')



if __name__ == "__main__":
    i = Indexer()           # instantiate an indexer
    q = SearchAgent(i)      # document retriever
    code.interact(local=dict(globals(), **locals())) # interactive shell