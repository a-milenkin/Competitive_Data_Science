from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2

stop_words = set(stopwords.words('russian'))

def text_prepare(text):
    lemmatizer = pymorphy2.MorphAnalyzer()
    word_tokens = word_tokenize(text)
    word_tokens = [w for w in word_tokens if not w in stop_words]
    word_tokens = [lemmatizer.parse(w)[0].normal_form for w in word_tokens]
    filtered_text = ' '.join(word_tokens)
    return filtered_text


def many_row_prepare(df, text_col='text'):
    res = []
    for text in df[text_col]:
        res.append(text_prepare(text))
    return res