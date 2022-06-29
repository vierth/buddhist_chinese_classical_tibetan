from pavdhutils import vector_analysis
from pavdhutils.general import load_text
from pavdhutils.cleaning import clean
from itertools import product
import time
import numpy as np

def get_seeds(text, size=2, ignore="一二三四五六七八九十百千"):
    seed_dict = {}
    for i in range(len(text)-(size-1)):
        seed = text[i:i+size]
        proceed = False
        for char in seed:
            if char not in ignore:
                proceed = True
                break

        if proceed:
            if seed in seed_dict:
                seed_dict[seed].append(i)
            else:
                seed_dict[seed] =[i]
    return seed_dict

def get_seed_vecs(text, model, size=4, seed_vecs={}):
    text_vecs = vector_analysis.text_to_word_vecs(text, model)
    for i in range(len(text)-(size-1)):
        
        seed = text[i:i+size]
        if seed not in seed_vecs:
            seed_vecs[seed] = vector_analysis.average_vectors(text_vecs[i:i+size])
        if i % 100000 == 0:
            print(f"{i} of {len(text)} analyzed {seed}")
    return seed_vecs

def get_shared_seeds(dict_1, dict_2, method="exact", threshold=.8, 
                    key_vectors={}):
    dict_1_keys = set(dict_1.keys())
    dict_2_keys = set(dict_2.keys())
    if method == "exact":
        return list(dict_1_keys.intersection(dict_2_keys))
    elif method == "vector similarity":
        valid_seed_pairs = []
        for key_1 in dict_1_keys:
            for key_2 in dict_2_keys:
                key_1_vec = key_vectors[key_1]
                key_2_vec = key_vectors[key_2]
                similarity = vector_analysis.compare_vectors([key_1_vec, key_2_vec])
                if similarity[0][1] > threshold:
                    print(key_1, key_2, similarity[0][1])
                    valid_seed_pairs.append((key_1, key_2))
        
        return valid_seed_pairs

def get_comparison_locations(source_dict, target_dict, matching_seeds):
    source_match_locations = []
    source_to_target_locations = {}
    for seed in matching_seeds:
        source_locations = source_dict[seed]
        target_locations = target_dict[seed]
        for source_location in source_locations:
            source_match_locations.append(source_location)
            source_to_target_locations[source_location] = target_locations
    return source_match_locations, source_to_target_locations


def compare_text_chunks(source_text_vecs, target_text_vecs, source_location, 
                        target_location, threshold=.9, min_length=10):
    source_vec = vector_analysis.average_vectors(source_text_vecs[source_location:source_location+min_length])
    target_vec = vector_analysis.average_vectors(target_text_vecs[target_location:target_location+min_length])
    similarity = vector_analysis.compare_vectors([source_vec, target_vec])
    if similarity[0][1] > threshold:
        new_source_start, new_target_start, new_similarity = optimize_start(source_text_vecs, 
                        target_text_vecs, source_location, target_location, model, 
                        threshold=threshold, min_length=min_length)
        #print(f"new attempt, starting sim = {similarity[0][1]}")
        
        new_source_len, new_target_len, new_similarity = optimize_match(source_text_vecs, 
                target_text_vecs, new_source_start, new_target_start, 
                min_length, new_similarity)
        return new_source_start, new_target_start, new_source_len, new_similarity
    else:
        return None

# mingshi = clean(load_text('D:/corpora/vierthcorpus/texts/754.txt', simplified=True))[:100000]
# jinshi = clean(load_text('D:/corpora/vierthcorpus/texts/751.txt', simplified=True))[:100000]

def optimize_match(source_text_vecs, target_text_vecs, source_loc, target_loc,
                    length, similarity, threshold=.9):
    source_length = length
    target_length = length
    prev_similarity = similarity
    similarity_drops = 0
    last_up_state = (source_length, target_length, similarity)
    while similarity > threshold and similarity_drops < 10:
        if similarity < prev_similarity:
            similarity_drops += 1
        elif similarity > prev_similarity:
            similarity_drops = 0
            last_up_state = (source_length, target_length, similarity)
        prev_similarity = similarity
        # source_up = vector_analysis.compare_vectors(
        #     [vector_analysis.average_vectors(source_text_vecs[source_loc:source_loc+source_length+1]),
        #     vector_analysis.average_vectors(target_text_vecs[target_loc:target_loc+target_length])])[0][1]
        # target_up = vector_analysis.compare_vectors(
        #     [vector_analysis.average_vectors(source_text_vecs[source_loc:source_loc+source_length]),
        #     vector_analysis.average_vectors(target_text_vecs[target_loc:target_loc+target_length+1])])[0][1]
        similarity = vector_analysis.compare_vectors(
            [vector_analysis.average_vectors(source_text_vecs[source_loc:source_loc+source_length+1]),
            vector_analysis.average_vectors(target_text_vecs[target_loc:target_loc+target_length+1])])[0][1]
        
      
        source_length += 1
        target_length += 1

    return last_up_state

def optimize_start(source_text_vecs, target_text_vecs, source_location, 
                    target_location, model, threshold=.9, min_length=10):
    new_source_start = source_location-min_length+2
    new_target_start = target_location-min_length+2
    if new_source_start < 0:
        new_source_start = 0
    if new_target_start < 0:
        new_target_start = 0

    source_chunks = [(vector_analysis.average_vectors(source_text_vecs[i:i+min_length]),i) for i in range(new_source_start,source_location+1)]
    target_chunks = [(vector_analysis.average_vectors(source_text_vecs[i:i+min_length]),i) for i in range(new_target_start,target_location+1)]
    text_pairs = [r for r in product(source_chunks, target_chunks)]
    # pairwise = []
    # for r in text_pairs:
    #     comparison = vector_analysis.compare_vectors([r[0][0], r[1][0]])
    #     #print(comparison, r[0], r[1])
    #     pairwise.append(comparison[0][1])
    pairwise = [vector_analysis.compare_vectors([r[0][0], r[1][0]])[0][1] for r in text_pairs]
    max_location = pairwise.index(max(pairwise))
    new_source_start, new_target_start = text_pairs[max_location][0][1], text_pairs[max_location][1][1]

    return new_source_start, new_target_start, max(pairwise)

def align_sequence(sequence_1_vecs, sequence_2_vecs, scoring_matrix={},gap_score=.7):
    #scores = vector_analysis.compare_vectors(sequence_1_vecs,extra={"Y":sequence_2_vecs})
    matrix = np.zeros([len(sequence_1_vecs)+1, len(sequence_2_vecs)+1])
    for i in range(len(sequence_1_vecs)+1):
        matrix[i][0] = -i*gap_score
    for j in range(len(sequence_2_vecs)+1):
        matrix[0][j] = -i*gap_score

    for i in range(len(sequence_1_vecs)):
        for j in range(len(sequence_2_vecs)):
            one_vec = sequence_1_vecs[i]
            two_vec = sequence_2_vecs[j]
            score = vector_analysis.compare_vectors([one_vec, two_vec])[0][1]
            print(score)
            matrixrow = i+1
            matrixcolumn = j+1

            # Calculate scores from top, left, and diagnol
            upperscore = matrix[i][j+1] + score
            leftscore = matrix[i+1][j] + score
            diagonal = matrix[i][j] + score

            # Select the highest score and place it in the box
            currentscore = max([upperscore, leftscore, diagonal])
            matrix[matrixrow][matrixcolumn] = currentscore

    # sentence order
    sentences_1 = []
    sentences_2 = []

    # traceback
    i = len(matrix)-1
    j = len(matrix[0])-1

    finalscore = matrix[i][j]

    # While i or j is above zero, trace backwards
    while i > 0 and j > 0:
        # Get the maximum value from the adjacent squares
        upper = matrix[i][j - 1]
        left  = matrix[i-1][j]
        diagonal = matrix[i-1][j-1]
        maxval = max([upper, left, diagonal]) 

        # If the maximum value is the diagonal, move diagonally
        if maxval == diagonal:
            i -= 1
            j -= 1
            sentences_1.append(i)
            sentences_2.append(j)
        # If the maximum value is above, insert gap into stringa    
        elif maxval == upper:
            j -= 1
            sentences_1.append(-1)
            sentences_2.append(j)

        # If the maximum value is left, insert gap into stringb
        elif maxval == left:
            i -= 1
            sentences_1.append(i)
            sentences_2.append(-1)
        
            
    return sentences_1[::-1], sentences_2[::-1]

def sm(a, b, scores):
    score = a[i], b[j]


if __name__ == "__main__":
    model = vector_analysis.load_word2vec("word2vec.model", verbose=True)

    mingshi = clean(load_text('D:/corpora/vierthcorpus/texts/754.txt', simplified=True))
    jinshi = clean(load_text('D:/corpora/vierthcorpus/texts/751.txt', simplified=True))

    mingshivecs = vector_analysis.text_to_word_vecs(mingshi, model)[:10000]
    jinshivecs =  vector_analysis.text_to_word_vecs(jinshi, model)[:10000]

    align_sequence(mingshivecs, jinshivecs)

    # res = get_seeds(mingshi)
    # res2 = get_seeds(jinshi)

    # # key_vecs = get_seed_vecs(mingshi, model)
    # # key_vecs = get_seed_vecs(jinshi, model, seed_vecs=key_vecs)


    # shared_seeds = get_shared_seeds(res, res2, method='exact')

    # source_locations, target_location_dict = get_comparison_locations(res, res2, shared_seeds)
    # last_t = time.time()
    # all_results = []
    # mapped_locations = {}
    # i = 0
    # while i < len(source_locations):
    #     source_location = source_locations[i]
    #     if i % 100 == 0:
    #         current_t = time.time()
    #         print(i, len(source_locations), f"{current_t - last_t:.2f}")
    #         last_t = current_t
    #     target_locations = target_location_dict[source_location]
    #     j = 0
    #     while j < len(target_locations):
    #         target_location = target_locations[j]
    #         run_analysis = True
    #         if source_location in mapped_locations:
    #             if target_location in mapped_locations[source_location]:
    #                 run_analysis = False
    #         if run_analysis:
    #             results = compare_text_chunks(mingshivecs, jinshivecs, source_location, target_location)
                
    #             if results:
    #                 source_loc, target_loc, length, sim  = results
    #                 all_results.append("\t".join([str(item) for item in results]))
    #                 for s_l, t_l in zip(range(source_loc, source_loc+length),range(target_loc, target_loc+length)):
    #                     if s_l not in mapped_locations:
    #                         mapped_locations[s_l] = [t_l]
    #                     else:
    #                         mapped_locations[s_l].append([t_l])
            
    #         j += 1
        
    #     i += 1

    # with open("simres.tsv",'w', encoding='utf8') as wf:
    #     wf.write("\n".join(all_results))