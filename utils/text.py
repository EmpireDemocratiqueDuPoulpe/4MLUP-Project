# #################################################################################################################### #
#       text.py                                                                                                        #
#           Transform text.                                                                                            #
# #################################################################################################################### #

import re
import string
import pandas
from unidecode import unidecode
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean(text):
    text = text.lower()  # Lower case
    text = re.sub(r"\W+", " ", text)  # Remove special characters
    text = re.sub(r"\d", "", text)  # Remove numbers
    text = unidecode(unidecode(text, "utf-8"))  # Remove accents
    text = re.sub(r"\s+", " ", text)  # Replace multi-space by single one
    text = re.sub(r"^\s/\s$", "", text)  # Trim space at beginning and end
    text = re.sub(r"\s+[a-zA-Z]\s+", "", text)  # Remove single characters

    return text


def remove_stopwords(token, language="english"):
    s_word = stopwords.words(language)
    return [word for word in token if token not in s_word]


def remove_punctuation(token):
    return [word for word in token if word not in string.punctuation]


def tokenize(text, language="english", preserve_line=False):
    return word_tokenize(text, language=language, preserve_line=preserve_line)


def stem(token):
    return [PorterStemmer().stem(word) for word in token]


def lemmatize(token):
    return [WordNetLemmatizer().lemmatize(word) for word in token]


def vectorization(data, col, analyzer, max_features=None):
    vectorized = CountVectorizer(analyzer=analyzer, max_features=max_features)
    x = vectorized.fit_transform(data[col])

    data_text = pandas.DataFrame(x.toarray())
    data_text.columns = vectorized.get_feature_names_out()

    return data_text


def tfidf_vectorization(data, col, analyzer, max_features=None):
    tfidf_vectorized = TfidfVectorizer(analyzer=analyzer, max_features=max_features)
    x = tfidf_vectorized.fit_transform(data[col])

    data_text = pandas.DataFrame(x.toarray())
    data_text.columns = tfidf_vectorized.get_feature_names_out()

    return data_text
