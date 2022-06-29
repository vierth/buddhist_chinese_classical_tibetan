import fasttext, os, pickle
from pavdhutils.general import load_csv, load_text
from scipy import spatial
import pandas as pd
from pavdhutils.vector_analysis import average_vectors, compare_vectors
from pavdhutils.tokenize1 import dict_tokenize
from pavdhutils.intertextuality import align_sequence
import seaborn as sns
import numpy as np
from dict_tokenize import Tokenizer

class Document:
    """
    Class to store and analyze documents
    """
    def __init__(self, text, sent_split="\n" , remove=None):
        if remove:
            while remove in text:
                text = text.replace(remove, " ")
        self.text = text
        self.sents = self.text.split(sent_split)
        self.sent_words = [s.split(" ") for s in self.sents]
        self.all_words = []
        self.all_words.extend([word for sent in self.sent_words for word in sent])

    def get_embeddings(self, embeddings):
        self.sent_word_vecs = []
        self.word_vecs = []
        self.used_words = []
        for sent in self.sent_words:
            sent_word_vec = [embeddings[word] for word in sent if word in embeddings and word != ""]
            used_words = [word for word in sent if word in embeddings and word != ""]
            if len(sent_word_vec) > 0:
                self.sent_word_vecs.append(sent_word_vec)
                self.word_vecs.extend(sent_word_vec)
                self.used_words.extend(used_words)

    def get_sent_vec(self):
        self.sent_vecs = [average_vectors(sent) for sent in self.sent_word_vecs]

    def get_doc_vec(self, vec_method="sentences"):
        if not hasattr(self,"sent_vecs"):
            self.get_sent_vec()
        if vec_method == "sentences":
            self.doc_vec = average_vectors(self.sent_vecs)
        elif vec_method == "words":
            self.doc_vec = average_vectors(self.word_vecs)

    def get_doc_cosine_similarity(self, other, vec_method="sentences"):
        if not hasattr(self, 'doc_vec'):
            self.get_doc_vec(vec_method=vec_method)
        if not hasattr(other, 'doc_vec'):
            other.get_doc_vec(vec_method=vec_method)

        return compare_vectors([self.doc_vec, other.doc_vec])[0][1]

    def get_doc_distance(self, other, vec_method="sentences"):
        if not hasattr(self, 'doc_vec'):
            self.get_doc_vec(vec_method=vec_method)
        if not hasattr(other, 'doc_vec'):
            other.get_doc_vec(vec_method=vec_method)

        return compare_vectors([self.doc_vec, other.doc_vec], method="euclidean distance")[0][1]

    def align_vectors(self, other, align_level="sentences", gap_score=.2):
        if align_level == "sentences":
            return align_sequence(self.sent_vecs, other.sent_vecs, gap_score=gap_score)
        elif align_level == "words":
            return align_sequence(self.word_vecs, other.word_vecs, gap_score=gap_score)

def compile_text(folder, outputfile, format='text'):
    print(f'compiling {folder} into {outputfile}')
    documents = []
    if format == "pickle":
        for root, dirs, files in os.walk(folder):
            for fname in files:
                with open(os.path.join(root, fname),'rb') as rf:
                    current_document = pickle.load(rf)

                    # PV using new tokenizer to tokenize corpus
                    current_document = [reparse_words(sent) for sent in current_document]
                    current_document = [" ".join(sent) for sent in current_document]
                    current_document = "\n".join(current_document)
                    documents.append(current_document)

    elif format == "text":
        for root, dirs, files in os.walk(folder):
            for fname in files:
                with open(os.path.join(root, fname),'r', encoding='utf8') as rf:
                    text = rf.read()
                    documents.append(text)

    with open(outputfile, 'w', encoding='utf8') as wf:
        wf.write("\n".join(documents))

tokenizer = Tokenizer()

def reparse_words(list_of_words):
    input_string = "".join(list_of_words)
    return tokenizer.tokenize_string(input_string)


def create_embedding(corpusfile, outputfile, lang='chinese', word_list_path=None):
    print(f"creating embedding for {corpusfile}")
    
    if lang == "chinese":
        model = fasttext.train_unsupervised(corpusfile, minCount=1, minn=1, maxn=10)
    else:
        model = fasttext.train_unsupervised(corpusfile, minCount=1)
    model.save_model(f'{outputfile}.bin')

    #https://stackoverflow.com/questions/58337469/how-to-save-fasttext-model-in-vec-format
    words = model.get_words()
    with open(f"{outputfile}.vec",'w',encoding='utf8') as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")
        # line by line, you append vectors to VEC file
        word_dict = {}
        for w in words:
            v = model.get_word_vector(w)
            word_dict[w] = v
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr+'\n')
            except:
                pass

        if word_list_path:
            word_list = [l[0] for l in load_csv(word_list_path)]
            for w in word_list:
                if len(w) > 1:
                    word_vec = [word_dict[c] for c in w if c in word_dict]
                    v = average_vectors(word_vec)
                v = model.get_word_vector(w)
                vstr = ""
                for vi in v:
                    vstr += " " + str(vi)
                try:
                    file_out.write(w + vstr+'\n')
                except:
                    pass

def check_embedding_coverage(corpusfile, embedding_file):
    print(f"checking {embedding_file} coverage of {corpusfile}")
    vecs = load_csv(embedding_file, delim=" ")
    vec_words = set([v[0] for v in vecs if v != ""])
    text = load_text(corpusfile).split("\n")
    corpus_words = []
    for line in text:
        words = line.split(" ")
        corpus_words.extend(words)
    corpus_words = set(corpus_words)
    coverage = corpus_words.intersection(vec_words)
    print(f"Of {len(corpus_words)} unique words, {len(coverage)} have embeddings")

def get_embedding_pairs(vec_1_file, vec_2_file, pair_file, outputfile, flip=False, level='word'):
    vec_1 = load_csv(vec_1_file, delim=" ")
    vec_2 = load_csv(vec_2_file, delim=" ")
    pairs = load_csv(pair_file, delim=",")
    # temp limit to just single char items This was not particularly effective, so skiping
    if level == "char":
        pairs = [p for p in pairs if len(p[0]) == 1]
    vec_1_words = set([v1[0] for v1 in vec_1 if v1 != ""])
    vec_2_words = set([v2[0] for v2 in vec_2 if v2 != ""])


    matched_embeddings = []

    for word_pair in pairs:
        if word_pair[0] in vec_1_words and word_pair[1] in vec_2_words:
            if not flip:
                matched_embeddings.append(",".join(word_pair))
            else:
                matched_embeddings.append(",".join(word_pair[::-1]))
    with open(outputfile,'w', encoding='utf8') as wf:
        wf.write("\n".join(matched_embeddings))



def combine_embeddings(vec_1, vec_2, matched_embeddings, out_1, out_2):
    """
    This runs the Glavas team code
    """
    os.system(f"python tm_trainer.py {vec_1} {vec_2} {matched_embeddings} {out_1} {out_2}")
    
def compare_embeddings(vec_1, vec_2, matched_embeddings, outputfile):
    word_pairs = {}
    all_paired_words = set()
    with open(matched_embeddings, 'r', encoding='utf8') as rf:
        data = rf.read().split("\n")
        for line in data:
            if line != "":
                d = line.split(",")
                word_pairs[d[0]] = d[1]
                all_paired_words.add(d[0])
                all_paired_words.add(d[1])


    print("loaded word pairs")
    print("loading language 1 embeddings")

    with open(vec_1, 'rb') as rf:
        vec_1_embedding = pickle.load(rf)

    print("loaded language 1 embeddings")
    print("loading language 2 embeddings")
    with open(vec_2,'rb') as rf:
        vec_2_embedding = pickle.load(rf)

    print("loaded language 2 embeddings")
    

    print("checking similarities")
    results = []
    results_num = []

    word_pairs_items = list(word_pairs.items())
    for key, val in word_pairs_items:

        vec_1_word = vec_1_embedding[key]
        vec_2_word = vec_2_embedding[val]
        results.append((key, val, str(1-spatial.distance.cosine(vec_1_word,vec_2_word))))
        results_num.append((key, val, 1-spatial.distance.cosine(vec_1_word,vec_2_word)))


    df = pd.DataFrame(results_num, columns=["Lang 1", "Lang 2", "similarity"])
    df = df.sort_values("similarity", ascending=False)

    print(["mean", df['similarity'].mean(), "std", df['similarity'].std(), "max", df['similarity'].max(), "min", df['similarity'].min()])

    with open(outputfile, 'w', encoding='utf8') as wf:
        sorted_res = sorted(results, key=lambda x:x[2], reverse=True)
        res = [",".join(r) for r in sorted_res]
        wf.write("\n".join(res))

def load_single_embeds(file_path):
    with open(file_path, 'r', encoding='utf8') as rf:
        lines = rf.read().split("\n")[1:]
        lines = [l.split(" ") for l in lines if l != ""]
    return {l[0]:np.asfarray(l[1:]) for l in lines}


def load_embeddings(file_path_1, file_path_2):
    print("loading embedding files")
    with open(file_path_1, 'rb') as rf:
        embed_1 = pickle.load(rf)
    with open(file_path_2, 'rb') as rf:
        embed_2 = pickle.load(rf)

    print("combining embeddings")
    return {**embed_1, **embed_2}


def data_creation_char():
    compile_text('processed_docs_char', 'chinese_corpus.txt', format='pickle')
    create_embedding("chinese_corpus.txt", "chinese_model", lang='chinese', word_list_path="wordpairs.txt")

def data_creation_word():
    # compile_text('reparse', 'chinese_corpus.txt', format='pickle')
    create_embedding("chinese_corpus.txt", "chinese_model_word", lang='chinese')

def data_creation_term():
    compile_text('reparse', 'chinese_corpus_2.txt', format='pickle')
    create_embedding("chinese_corpus_2.txt", "chinese_model_hybrid_char_term_2", lang='chinese')


if __name__ == "__main__":
    # if embeddings need to be created
    # create chinese embeddings

    # this uses a character based embedding
    # data_creation_char()
    # check_embedding_coverage('chinese_corpus.txt', 'chinese_model.vec')
    
    # data_creation_word()
    # data_creation_term()
    # create tibetan embeddings if needed
    # compile_text('segall', 'tibetan_corpus.txt', format='text')
    # create_embedding('tibetan_corpus.txt', 'tibetan_model')
    # remember to always remove (་ )
    # check_embedding_coverage('tibetan_corpus.txt', 'tib_seg.vec')

    # get paired embeddings
    # get_embedding_pairs('chinese_model.vec', 'tib_seg.vec', 'wordpairs.txt', 'matched_embeddings_char.txt', level="char", flip=True)
    
    # projecting the Chinese into the Tibetan space seems to work better, so you can do so here
    get_embedding_pairs('chinese_model_hybrid_char_term_2.vec', 'tib_seg.vec', 'wordpairs.txt', 'matched_embeddings_term2.txt', flip=True)
    # get_embedding_pairs('chinese_model_hybrid_char_term.vec', 'tib_seg.vec', 'wordpairs.txt', 'matched_embeddings_term.txt', flip=True)
    # get_embedding_pairs('chinese_model_word.vec', 'tib_seg.vec', 'wordpairs.txt', 'matched_embeddings_word.txt', flip=True)


    # combine embedding space tib to chinese
    # combine_embeddings("chinese_model.vec", 'tib_seg.vec', 'matched_embeddings.txt', 'chin_comb_char.p', 'tib_comb_char.p')

    # combine embedding space chinese to tib
    combine_embeddings('tib_seg.vec', "chinese_model_hybrid_char_term_2.vec",  'matched_embeddings_term2.txt', 'tib_comb_hybrid_term_2t.p', 'chin_comb_hybrid_term_2t.p')
    # combine_embeddings('tib_seg.vec', "chinese_model_word.vec",  'matched_embeddings_word.txt', 'tib_comb_word.p', 'chin_comb_word.p')

    # compare the resulting embeddings
    #compare_embeddings('chin_comb.p', 'tib_comb.p', 'matched_embeddings.txt', 'embed_sim.txt')
    #compare_embeddings('tib_comb_word.p', 'chin_comb_word.p', 'matched_embeddings_word.txt', 'embed_sim_word.txt')
    #compare_embeddings('mapped.t.vectors', 'mapped.c.vectors', 'matched_embeddings.txt', 'embed_sim.txt', instamap_dicts=['t.vocab', 'c.vocab'])


    # run doc align
    #aling_doc_sequences()