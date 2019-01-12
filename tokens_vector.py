# -*- coding: utf-8 -*-

from constants import *
from sklearn.feature_extraction.text import TfidfVectorizer, sp

vectorizer = TfidfVectorizer()


def generateVectors(path, npz_path):

    file = open(path, 'r')

    index = 0

    print('reading ......')

    tokens = []

    for line in file:

        words = line.strip().split('\t')
        token = words[1].split('|')
        tokens.append(' '.join(token))

        index += 1
        if index % 100000 == 0:
            print(index)
            # break

    file.close()
    print('fitting ......')
    tokens_vectors = vectorizer.fit_transform(tokens)
    print(vectorizer.get_feature_names())
    print(tokens_vectors.shape)
    print('saving %s ......' % tokens_vectors)
    sp.save_npz(npz_path, tokens_vectors)


if __name__ == '__main__':
    # tokens_vector_title = sp.load_npz(vectors_titleid)
    generateVectors(PATH_QUERY_ID, PATH_VEC_QUERY)
    generateVectors(PATH_TITLE_ID, PATH_VEC_TITLE)
    generateVectors(PATH_DESCRIPTION_ID, PATH_VEC_DESCRIPTION)
    generateVectors(PATH_KEYWORD_ID, PATH_VEC_KEYWORD)


