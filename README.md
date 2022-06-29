# Code for "Crosslinguistic Semantic Textual Similarit of Buddhist Chinese and Classical Tibetan
This set of scripts and materials creates an embedding space for Buddhist Chinese materials and projects these Chinese embeddings into the embeddings provided by Marieke. Work in progress, lots to come!

## embedding_pipeline.py
This script contains most of the code needed to create the Chinese embeddings and to project the embeddings into a shared space.

Do note that if you run this code and overwrite any of the pickle files in the repository then your code will not result in the exact same results as our output.

## crosslingual_search.py
This script runs the search algorithm. The constants at the beginning of the file allow you to run the search with different parameters.

## chinese_corpus_2.txt
This the Chinese corpus used to training the embedding model for the Chinese Hybrid terms 2 embedding. This is derived from the Kanseki repository and is segmented using the scripts included here.

## wordpairs.txt
This is the translation glossary used to project the embedding spaces together.

## Dependencies
sklearn, scipy, pandas, matplotlib, seaborn, fasttext (and for the external code, tensorflow)

## Glavas code 
You will need to seperately download code from [Glavas code](https://bitbucket.org/gg42554/cl-sts/src/master/code/) and make a few edits. I am not directly providing this code because it is not clear what license it uses.

Download and save io_helper.py, translation_matrix.py, and tm_trainer.py in the same directory as the code in this repository

### Edits to tm_trainer.py
Be sure to add this line tot he imports
```
import tensorflow.compat.v1 as tfc
```
and change references to `tf` to `tfc`

Replace lines 78 and 79 in tm_trainer.py with this:
```
len_emb_src = len(emb_dict_src[list(emb_dict_src.keys())[0]])
len_emb_trg = len(emb_dict_trg[list(emb_dict_trg.keys())[0]])
```

Replace lines 89 and 90 with
```
train_src = np.array([v[2] for v in train_set])
train_trg = np.array([v[3] for v in train_set])
```

### Edits to translation_matrix.py
Be sure to add this to imports
```
import tensorflow.compat.v1 as tfc
```
And call this before the rest of the code:
```
tfc.disable_eager_execution()
```

Finally, you will need to replace the init of TranslationMatrix with this
```
def __init__(self, emblen_first, emblen_sec):
		
    self.source_vectors = tfc.placeholder(tf.float32, shape = [None, emblen_first])
    self.target_vectors = tfc.placeholder(tf.float32, shape = [None, emblen_sec])
    
    self.trans_matrix = tf.Variable(tfc.random_uniform([emblen_first, emblen_sec], -1.0, 1.0))
    self.predictions_target = tf.matmul(self.source_vectors, self.trans_matrix)
    self.objective =  tf.nn.l2_loss(tf.subtract(self.predictions_target, self.target_vectors))
    
    self.train_op = tfc.train.AdamOptimizer(0.001).minimize(self.objective)
```
This simply ensures that the code will run on newer versions of tensorflow

