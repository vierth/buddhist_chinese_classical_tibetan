# Open philology align
This set of scripts creates an embedding space for Chinese materials and projects the Chinese embeddings into the embeddings provided by Marieke. Work in progress, lots to come!

## embedding_pipeline.py
Create embedding space, project to Tibetan (using the external scripts written by the Glavas team), and then evaluate against rafal's alignments

## Plans
Smith waterman to find local alignments of similar sentences. Scoring based on cosine similarity, but perhaps later based on supervised algo trained on team's alignments

## Dependencies
sklearn, scipy, pandas, matplotlib, seaborn, fasttext (and for the external code, tensorflow)