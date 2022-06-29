from gensim.models import Word2Vec
from gensim.models import FastText
# from pavdhutils.corpus import load_corpus
import os
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# creates a word2vec model, saves it, and returns it
# def create_word2vec(corpuslocation="D:/corpora/vierthcorpus/texts", 
#                     corpus_pickle='limited_corpus.p', included_texts=[],
#                     excluded_texts=[], save_path="word2vec.model", min_count=1, 
#                     workers=20, model_creation_args={}):
#     corpus_contents = load_corpus(corpus_path=corpuslocation, verbose=True,
#                                     include_list=included_texts,
#                                     exclude_list=excluded_texts, 
#                                     pickle_path=corpus_pickle)
#     corpus_sentences = [s for s in corpus_contents.values()]
#     model = Word2Vec(sentences=corpus_sentences, min_count=min_count, 
#                     workers=workers, **model_creation_args)
#     model.save(save_path)
#     return model

# Load a Word2Vec model
def load_word2vec(model_path, verbose=False):
    if verbose:
        print(f"Loading model from {model_path}")
    model = Word2Vec.load(model_path)
    if verbose:
        print(f"Model loaded")
    return model

def text_to_word_vecs(text, vec_model):
    return [vec_model.wv[word] for word in text if word in vec_model]

def average_vectors(vecs):
    return np.add.reduce(vecs)/len(vecs)

def add_vectors(vecs):
    return np.add.reduce(vecs)

'''
Creat a vector to represent a chunk of text.
'''
def text_vector(text, vec_model, method="average", verbose=False):
    if verbose:
        print("extracting word vectors")
    word_vectors = text_to_word_vecs(text, vec_model)

    if method == "average":
        if verbose:
            print(f"averaging {len(word_vectors)} vectors")
        return average_vectors(word_vectors)
    else:
        print(f"{method} not implemented")
        return None

'''
A function to compare vectors based on a variety of metrics.
'''
def compare_vectors(vector_list, method="cosine similarity", extra={}):
    if method == "cosine similarity":
        return cosine_similarity(vector_list, **extra)
    elif method == "linear kernel":
        return linear_kernel(vector_list, **extra)
    elif method == "euclidean distance":
        return pairwise_distances(vector_list, metric="euclidean", **extra)
    elif method == "cosine distance":
        return pairwise_distances(vector_list, metric="cosine", **extra)
    else:
        print(f"{method} not known")

def text_compare(text_1, text_2, vec_model, compare_method="cosine similarity", 
                extra={}):
    word_vectors_1 = average_vectors(text_to_word_vecs(text_1, vec_model))
    word_vectors_2 = average_vectors(text_to_word_vecs(text_2, vec_model))
    return compare_vectors([word_vectors_1, word_vectors_2],
                            method=compare_method, extra=extra)

def corpus_vectorizer(text_list, model, method="average"):
    text_vecs = [text_vector(text, model,method) for text in text_list]
    return text_vecs

if __name__ == "__main__":
    from pavdhutils.general import load_text
    from pavdhutils.cleaning import clean

    # excluded_range = [str(i)+".txt" for i in range(15, 46031)]
    # create_word2vec(excluded_texts=excluded_range, save_path="word2vec.model",
    #             model_creation_args={"size":200})
    model = load_word2vec("word2vec.model", verbose=True)
    mingshi = load_text('D:/corpora/vierthcorpus/texts/754.txt', simplified=True)
    jinshi = load_text('D:/corpora/vierthcorpus/texts/751.txt', simplified=True)
    yuanshi = load_text('D:/corpora/vierthcorpus/texts/752.txt', simplified=True)
    zhouyi = load_text('D:/corpora/vierthcorpus/texts/0.txt', simplified=True)

    texts = [mingshi, jinshi, yuanshi,zhouyi]
    res_vectors = corpus_vectorizer(texts,model)
    
    similarities = compare_vectors(res_vectors,method='cosine distance')
    print(similarities)