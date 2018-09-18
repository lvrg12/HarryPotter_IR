from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# tokenization and filtration of document
def tokenize( doc ):

    # tokenizing
    tokenized = word_tokenize(doc)

    # remove punctuations
    tokenized = [ word for word in tokenized if word.isalpha() ]

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered = [ w for w in tokenized if not w in stop_words ]
    filtered = []

    for w in tokenized:
        if w not in stop_words:
            filtered.append(w)

    return filtered

#lemmatization of document
def lemmatize( tokenized ):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [ lemmatizer.lemmatize(word) for word in tokenized ]
    return lemmatized

#stemming of document
def stem( tokenized ):
    porter = PorterStemmer()
    stemmed = [ porter.stem(word) for word in tokenized ]
    return stemmed