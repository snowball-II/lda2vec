from spacy.lang.en import English
from spacy.attrs import LOWER, LIKE_URL, LIKE_EMAIL
from tqdm import tqdm

import numpy as np


def tokenize(texts, max_length, skip=-2, attr=LOWER, merge=False, nlp=None,
             **kwargs):
    """ Uses spaCy to quickly tokenize text and return an array
    of indices.

    This method stores a global NLP directory in memory, and takes
    up to a minute to run for the time. Later calls will have the
    tokenizer in memory.

    Parameters
    ----------
    text : list of unicode strings
        These are the input documents. There can be multiple sentences per
        item in the list.
    max_length : int
        This is the maximum number of words per document. If the document is
        shorter then this number it will be padded to this length.
    skip : int, optional
        Short documents will be padded with this variable up until max_length.
    attr : int, from spacy.attrs
        What to transform the token to. Choice must be in spacy.attrs, and =
        common choices are (LOWER, LEMMA)
    merge : int, optional
        Merge noun phrases into a single token. Useful for turning 'New York'
        into a single token.
    nlp : None
        A spaCy NLP object. Useful for not reinstantiating the object multiple
        times.
    kwargs : dict, optional
        Any further argument will be sent to the spaCy tokenizer. For extra
        speed consider setting tag=False, parse=False, entity=False, or
        n_threads=8.

    Returns
    -------
    arr : 2D array of ints
        Has shape (len(texts), max_length). Each value represents
        the word index.
    vocab : dict
        Keys are the word index, and values are the string. The pad index gets
        mapped to None

    >>> sents = [u"Do you recall a class action lawsuit", u"hello zombo.com"]
    >>> arr, vocab = tokenize(sents, 10, merge=True)
    >>> arr.shape[0]
    2
    >>> arr.shape[1]
    10
    >>> w2i = {w: i for i, w in vocab.iteritems()}
    >>> arr[0, 0] == w2i[u'do']  # First word and its index should match
    True
    >>> arr[0, 1] == w2i[u'you']
    True
    >>> arr[0, -1]  # last word in 0th document is a pad word
    -2
    >>> arr[0, 4] == w2i[u'class action lawsuit']  # noun phrase is tokenized
    True
    >>> arr[1, 1]  # The URL token is thrown out
    -2
    """
    if nlp is None:
        nlp = English()

    data = np.zeros((len(texts), max_length), dtype='int64')
    data[:] = skip
    bad_deps = ('amod', 'compound')
    vocab = {}
    vocab_inverted = {}
    for row, doc in tqdm(enumerate(nlp.pipe(texts, **kwargs)), unit=''):
        for col, token in enumerate(doc):
            if col >= max_length:
                break
            if token.text not in vocab_inverted:
                vocab_inverted[token.text] = [len(vocab_inverted), 1]
                vocab[len(vocab)] = token.text
            else:
                vocab_inverted[token.text][1] += 1
            data[row, col] = vocab_inverted[token.text][0]
    # vocab[skip] = '<SKIP>'
    return data, vocab


if __name__ == "__main__":
    import doctest
    doctest.testmod()
