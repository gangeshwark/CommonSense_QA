import io
import operator
import re
import string
from pprint import pprint

import math
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import json
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


class Similarity():
    def __init__(self):
        self.vocab = []
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.input_para = "I went into my bedroom and flipped the light switch. Oh, I see that the ceiling lamp is not turning on. It must be that the light bulb needs replacement. I go through my closet and find a new light bulb that will fit this lamp and place it in my pocket. I also get my stepladder and place it under the lamp. I make sure the light switch is in the off position. I climb up the ladder and unscrew the old light bulb. I place the old bulb in my pocket and take out the new one. I then screw in the new bulb. I climb down the stepladder and place it back into the closet. I then throw out the old bulb into the recycling bin. I go back to my bedroom and turn on the light switch. I am happy to see that there is again light in my room."
        self.vocab = []
        # sw = stopwords.words('english')
        self.punc = string.punctuation
        data = self.read_json_data()

    def preprocess(self, text):
        texts = sent_tokenize(text)
        all_text = []
        # remove stop words
        for text in texts:
            # text = text.replace('.', '')
            text = re.findall(r"[\w']+|[.,!?;]", text)
            # print(len(text))
            text = [self.wordnet_lemmatizer.lemmatize(word.lower()) for word in text]
            text = [word.lower() for word in text if word and (word not in self.punc)]  # and (word.lower() not in sw)
            # print(len(text))
            all_text += text
        # print((len(set(all_text))))
        # print(sorted(list(set(all_text))))
        all_text = list(all_text)
        self.vocab += all_text
        return all_text

    def read_json_data(self, path='all_scripts.json'):
        all_scripts = {}
        with io.open('all_scripts.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for key in tqdm(sorted(data.keys())):
                scripts = data[key]
                for script in scripts:
                    all_scripts[key + '.' + script['id']] = self.preprocess(script['text'])
                    # print(all_scripts)
        return all_scripts

    def idf(self, data, vocab):
        idf_values = {}
        """
        for k, v in data.items():
            all_tokens_set += list(set([item for item in v]))
        all_tokens_set = set(all_tokens_set)
        """
        for tkn in tqdm(vocab):
            contains_token = 0
            for k, v in data.items():
                contains_token += v.count(tkn)
            # contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(data.keys()) / contains_token)
        return idf_values

    def jaccard_similarity(self, query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection) / len(union)

    def tf(self, term, tokenized_document):
        return tokenized_document.count(term)

    def sublinear_term_frequency(self, term, doc):
        count = doc.count(term)
        if count == 0:
            return 0
        return 1 + math.log(count)

    def augmented_term_frequency(self, term, doc):
        max_count = max([self.tf(t, doc) for t in doc])
        return 0.5 + ((0.5 * self.tf(term, doc)) / max_count)

    def tf_idf(self, data):
        idf_val = self.idf(data, self.vocab)
        tfidf_documents = {}
        for doc_id, doc in tqdm(data.items()):
            doc_tfidf = []
            for term in idf_val.keys():
                tf = self.sublinear_term_frequency(term, doc)
                doc_tfidf.append(tf * idf_val[term])
            tfidf_documents[doc_id] = doc_tfidf
        return tfidf_documents

    def cosine_similarity(self, vector1, vector2):
        dot_product = sum(p * q for p, q in zip(vector1, vector2))
        magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
        if not magnitude:
            return 0
        return dot_product / magnitude


if __name__ == '__main__':
    sim = Similarity()
