import io
import operator
import re
import string
import xml
from pprint import pprint

import math
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import json
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

wordnet_lemmatizer = WordNetLemmatizer()

input_para = "I went into my bedroom and flipped the light switch. Oh, I see that the ceiling lamp is not turning on. It must be that the light bulb needs replacement. I go through my closet and find a new light bulb that will fit this lamp and place it in my pocket. I also get my stepladder and place it under the lamp. I make sure the light switch is in the off position. I climb up the ladder and unscrew the old light bulb. I place the old bulb in my pocket and take out the new one. I then screw in the new bulb. I climb down the stepladder and place it back into the closet. I then throw out the old bulb into the recycling bin. I go back to my bedroom and turn on the light switch. I am happy to see that there is again light in my room."

vocab = []

# sw = stopwords.words('english')
punc = string.punctuation


def preprocess(text):
    global vocab
    texts = sent_tokenize(text)
    all_text = []
    # remove stop words
    for text in texts:
        # text = text.replace('.', '')
        text = re.findall(r"[\w']+|[.,!?;]", text)
        # print(len(text))
        text = [wordnet_lemmatizer.lemmatize(word.lower()) for word in text]
        text = [word.lower() for word in text if word and (word not in punc)]  # and (word.lower() not in sw)
        # print(len(text))
        all_text += text
    # print((len(set(all_text))))
    # print(sorted(list(set(all_text))))
    all_text = list(all_text)
    vocab += all_text

    return all_text


def read_json_data(path='all_scripts.json'):
    all_scripts = {}
    with io.open('all_scripts.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key in tqdm(sorted(data.keys())):
            scripts = data[key]
            for script in scripts:
                all_scripts[key + '.' + script['id']] = preprocess(script['text'])
                # print(all_scripts)
    return all_scripts


def idf(corpus_data, train_data, vocab):
    idf_values = {}
    """
    for k, v in data.items():
        all_tokens_set += list(set([item for item in v]))
    all_tokens_set = set(all_tokens_set)
    """
    for tkn in tqdm(vocab):
        contains_token = 0
        for k, v in corpus_data.items():
            contains_token += v.count(tkn)

        for k, v in train_data.items():
            contains_token += v.count(tkn)
        # contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        if contains_token == 0:
            contains_token = 1
        idf_values[tkn] = 1 + math.log((len(train_data.keys()) + len(corpus_data.keys())) / contains_token)
    return idf_values


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def tf(term, tokenized_document):
    return tokenized_document.count(term)


def sublinear_term_frequency(term, doc):
    count = doc.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)


def augmented_term_frequency(term, doc):
    max_count = max([tf(t, doc) for t in doc])
    return 0.5 + ((0.5 * tf(term, doc)) / max_count)


def tf_idf(corpus_data, train_data):
    idf_val = idf(corpus_data, train_data, vocab)
    tfidf_documents = {}
    for doc_id, doc in tqdm(corpus_data.items()):
        doc_tfidf = []
        for term in idf_val.keys():
            tf = sublinear_term_frequency(term, doc)
            doc_tfidf.append(tf * idf_val[term])
        tfidf_documents[doc_id] = doc_tfidf
    return tfidf_documents


def cosine_similarity(vector1, vector2):
    dot_product = sum(p * q for p, q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product / magnitude


def load_train_data(path='data/train-data.xml'):
    root = xml.etree.ElementTree.parse(path).getroot()
    """
    <data>
      <instance id="0">
        <text>I went into my bedroom and flipped the light switch. Oh, I see that the ceiling lamp is not turning on. It must be that the light bulb needs replacement. I go through my closet and find a new light bulb that will fit this lamp and place it in my pocket. I also get my stepladder and place it under the lamp. I make sure the light switch is in the off position. I climb up the ladder and unscrew the old light bulb. I place the old bulb in my pocket and take out the new one. I then screw in the new bulb. I climb down the stepladder and place it back into the closet. I then throw out the old bulb into the recycling bin. I go back to my bedroom and turn on the light switch. I am happy to see that there is again light in my room.</text>
        <questions>
          <question id="0" text="Which room did the light go out in?">
            <answer correct="False" id="0" text="Kitchen."/>
            <answer correct="True" id="1" text="Bedroom."/>
          </question>
        </questions>
      </instance>
    </data>
    """
    data = {}
    for instance in tqdm(root.findall('instance')):
        data[instance.get('id')] = preprocess(instance.find('text').text)
    return data


if __name__ == '__main__':
    corpus_data = read_json_data()
    train_data = load_train_data()
    vocab = set(vocab)
    print("Len of vocab: ", len(set(vocab)))
    print(jaccard_similarity(corpus_data['access_the_internet.xml.1'], corpus_data['baking a cake.new.xml.2']))
    # idf_val = idf(data, vocab)
    idf_val = {}
    print(len(idf_val.items()))
    with io.open('tdf_values.txt', 'w') as file:
        for key, value in sorted(idf_val.items(), key=operator.itemgetter(1), reverse=True):
            file.write("%s\t%s\n" % (key, value))
    tf_idf_feat = tf_idf(corpus_data, train_data)
    print(tf_idf_feat['access_the_internet.xml.1'])
    print(cosine_similarity(tf_idf_feat['access_the_internet.xml.1'], tf_idf_feat['baking a cake.new.xml.2']))
    print(cosine_similarity(tf_idf_feat['access_the_internet.xml.1'], tf_idf_feat['access_the_internesst.xml.1']))
