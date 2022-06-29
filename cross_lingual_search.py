from embedding_pipeline import load_embeddings
from pavdhutils.general import load_csv
from pavdhutils.vector_analysis import average_vectors
from pavdhutils.tokenize1 import dict_tokenize
from dict_tokenize import Tokenizer
import numpy as np
from scipy.spatial import distance
import os, statistics


# to tune for optimal results, give a tuple with start, end, and step.
# results will be recorded to file only if step is set to 0

TIB_ADJUST = 1 # factor to adjust the tibetan words by, as more words are used.
TIB_ADJUST_SHORT = 1.3  # longer adjustment factor for shorter phrases
SHORT_LENGTH = 20 # what constitutes "short"
CLUSTER_WINDOW = "full" # the length of results to use as cluster window. Should be integer or "full" to capture all overlap
TOTAL_RESULTS = 500
FULL_RESULT = .9 # Anything over this is considered a full result
PARTIAL_RESULT = .5 # Anything over this is considered a partial result
ANALYSIS_FILE = 'text_one.csv' # expects text_one.csv, text_two.csv, or text_three.csv 

# cosine is the only distance we have extensively tested
DISTANCE_METRIC = "cosine"



TOKENIZATION_METHOD = "term2"
REPORTING = "terse" # or verbose


# grab correct embedding models, and create a tokenizer if need be.
embedding_models = {
    "char": ['tib_comb.p','chin_comb.p', None],
    "word": ['tib_comb_word.p', 'chin_comb_word.p', None],
    "term": ['tib_comb_hybrid_term.p', 'chin_comb_hybrid_term.p', "karashima_terms.txt"],
    "term2": ['tib_comb_hybrid_term_2.p', 'chin_comb_hybrid_term_2.p',"karashima_and_dazhidulun.txt"]
}


# grab embedding info
TIB_EMBED = embedding_models[TOKENIZATION_METHOD][0]
CHIN_EMBED = embedding_models[TOKENIZATION_METHOD][1]
if embedding_models[TOKENIZATION_METHOD][2]:
    tokenizer = Tokenizer(embedding_models[TOKENIZATION_METHOD][2])


# describes where the data lives depending on input spreadsheet, given slightly different formats
data_info = {
    "text_one.csv": [None, 1, 2],
    "text_two.csv": [0, 1, 2],
    "text_three.csv": [0, 1, 2]
}

# grab correct settings depending on analysis file
RANK_IND = data_info[ANALYSIS_FILE][2]
TIB = data_info[ANALYSIS_FILE][0]
CHIN = data_info[ANALYSIS_FILE][1]

if ANALYSIS_FILE == "text_one.csv":
    TIB_FILE = "seg/D75-tibetan_unicode.txt"
    CORR_TIB = 0
    LEAVE_OUT = [2, 3, 21]
else:
    LEAVE_OUT = []

summary_results = []
all_lengths = []


def vectorize_list(items, embeddings):
    return [embeddings[item] for item in items if item in embeddings]

def vectorize_chunks(word_list, chunk_len, embeddings):
    return [average_vectors(vectorize_list(word_list[i:i+chunk_len], embeddings)) for i in range(len(word_list)-(chunk_len-1))]

def vectorize_phrase(phrase, embeddings, tokenization_method="char"):
    if tokenization_method == "char":
        return average_vectors([embeddings[w] for w in phrase if w in embeddings])
    elif tokenization_method == "word":
        phrase = dict_tokenize(phrase, as_string=" ")
        return average_vectors([embeddings[w] for w in phrase.split(" ") if w in embeddings])
    elif tokenization_method == "term" or tokenization_method == "term2":
        phrase = tokenizer.tokenize_string(phrase)
        phrase_vecs = []
        for w in phrase:
            if w in embeddings:
                phrase_vecs.append(embeddings[w])
            else:
                if len(w) > 1:
                    temp_vec = []
                    for sub_w in w:
                        if sub_w in embeddings:
                            temp_vec.append(embeddings[sub_w])
                    if len(temp_vec) > 0:
                        phrase_vecs.append(average_vectors(temp_vec))

        return average_vectors(phrase_vecs)

#https://stackoverflow.com/questions/53455909/python-optimized-most-cosine-similar-vector
def get_most_similar(search_term_vec, target_language_vectors):   
    distances = distance.cdist([search_term_vec], target_language_vectors, DISTANCE_METRIC)[0]   
    top_results = np.argsort(distances)
    results = [(res, distances[res]) for res in top_results]
    return results

def get_list_loc(search_term, full_list):
    found = False
    if search_term[0] in full_list:
        first_term_loc = full_list.index(search_term[0])
        if search_term == full_list[first_term_loc:first_term_loc + len(search_term)]:
            found = True
            return first_term_loc
        while not found:
            first_term_loc = full_list.index(search_term[0],first_term_loc+1)
            if search_term == full_list[first_term_loc:first_term_loc + len(search_term)]:
                found = True
                return first_term_loc
    else:
        return None
        
# run the analysis
def run(embeddings, write_output=True):
    if write_output:
        results_dir = f'{ANALYSIS_FILE[:-4]}_{CLUSTER_WINDOW}_{int(TIB_ADJUST_SHORT*100)}_{SHORT_LENGTH}_{int(TIB_ADJUST*100)}_{TOTAL_RESULTS}_{TOKENIZATION_METHOD}'


    # all ranks
    all_predicted_ranks = []
    
    
    # load the data to analyze
    data = load_csv(ANALYSIS_FILE)
    data = [d for d in data if len(d) > 0]

    # get the Chinese and Tibetan pharses from the data
    chinese = [d[CHIN] for d in data if d[RANK_IND] == "5"]

    if TIB:
        lim_tibetan  = [d[TIB].replace('་ ', ' ') for d in data if d[RANK_IND] == "5"]
    elif not TIB and type(CORR_TIB) == int:
        lim_tibetan = [d[CORR_TIB].replace('་ ', ' ') for d in data if d[RANK_IND] == "5"]

    if TIB:
        tibetan = [d[TIB] for d in data]
    else:
        with open(TIB_FILE, "r", encoding='utf8') as rf:
            tibetan = rf.read().split("<utt>\n")
            
    # set up summary dictionary
    summary_info = {"some_overlap": 0, 
                    "full_overlap": 0, 
                    "full_overlap_top_5":0, 
                    "top_res":0,
                    "some_top":0,
                    "some_top_5":0,
                    "top_full_overlap":[],
                    "top_any_overlap":[],
                    "total_phrases":len(chinese)-len(LEAVE_OUT)}



    # Combine the tibetan into one long text, remove sylable markers from end
    # of phrase (but not internally)
    tibetan_text = " ".join(tibetan).replace('་ ', ' ')
    
    if tibetan_text[-1] == '་':
        tibetan_text = tibetan_text[:-1]

    # turn into a list and remove empty items
    tibetan_words = tibetan_text.split(" ")
    
    tibetan_words = [t for t in tibetan_words if t != ""]


    # iterate through each Chinese term
    for i,search_term in enumerate(chinese):
        if i in LEAVE_OUT:
            continue

        # get correct location for tibetan for later analysis
        
        if TIB or (not TIB and type(CORR_TIB)==int):
            
            accurate_align = lim_tibetan[i].replace('་ ', ' ')
            
            if len(accurate_align) > 0:
                if accurate_align[-1] == "་":
                    accurate_align = accurate_align[:-1]
                correct_tib_loc = get_list_loc(accurate_align.split(" "), tibetan_words)
            else:
                correct_tib_loc = ""
        else:
            accurate_align = ""
            correct_tib_loc = ""


        # remove punctuation
        search_term = search_term.translate(str.maketrans('', '', "：。，」「；、 ？！『』"))
    
        # get the vectors for the search term
        search_term_vectors = vectorize_phrase(search_term, embeddings, TOKENIZATION_METHOD)

        all_lengths.append(len(search_term))

        # if the lenght of the search term is greater than threshold, use
        # standard adjustment
        if len(search_term) > SHORT_LENGTH:
            search_len = int(len(search_term) * (1+TIB_ADJUST))
            used_ratio = 1+TIB_ADJUST
        # if the lenght is shorter, use the short adjustment
        else:
            search_len = int(len(search_term) * (1+TIB_ADJUST_SHORT))
            used_ratio = 1+TIB_ADJUST_SHORT
        # set the window for clustering the results       
        cluster_window = CLUSTER_WINDOW
        if cluster_window == "full":
            cluster_window = search_len


        # get vectors for tibetan words depending on the search_len calculated
        # above
        search_chunks = vectorize_chunks(tibetan_words, search_len, embeddings)
        
        # get the most similar results for 
        
        results = get_most_similar(search_term_vectors, search_chunks)
        just_ind = [r[0] for r in results]
        all_loc_poss = len(just_ind)
        if correct_tib_loc:
            tib_result_loc = just_ind.index(correct_tib_loc)
        else:
            tib_result_loc = "None"

        # convert from distance to similarity ()
        num_res = [1-r[1] for r in results]

        # calculate some summary statistics
        std = np.std(num_res)
        mean_num = np.mean(num_res)
    
        # limit to the top N results
        results = results[:TOTAL_RESULTS]
        results = [[r[0], r[1], i+1, r[0]+search_len-1] for i,r in enumerate(results)]        


        # merge proximal ranges within a certain window
        sorted_results = sorted(results, key=lambda x:x[0])

        # dictionary to store a given result's similarity score and rank
        result_info = {}
        index_to_cosine = {}
        # iterate through sorted results:
        merged_ranges = []
        range_rank = []
        current_range = []
        max_rank = None
        for j,res in enumerate(sorted_results):
            # save the rank pointing to place and cosine sim
            result_info[res[2]] = [res[0], res[1]]
            index_to_cosine[res[0]] = res[1]
            if current_range == []:
                current_range = [res[0], res[3]]
                max_rank = res[2]
        
            else:
                # if the beginning of the current range is less than or equal to
                # end of last range (plus one for adjacent), set end of range
                if res[0] <= current_range[1] + cluster_window + 1:
                    current_range[1] = res[3]
                    if res[2] < max_rank:
                        max_rank = res[2]
                # otherwsie append to merged ranges and reset current range
                else:            
                    merged_ranges.append(current_range)
                    range_rank.append(max_rank)
                    current_range = [res[0], res[3]]
                    max_rank = res[2]
            if j == len(sorted_results) - 1:
                merged_ranges.append(current_range)
                range_rank.append(max_rank)

        merge_rank = [[r,rank] for rank,r in sorted(zip(range_rank, merged_ranges))]

        text_results = []

        # calculate which results overlap with actual tibetan and by how much
        overlap_with_actual = {}

        for j,item in enumerate(merge_rank):
            active_range = item[0]
            rank = item[1]
            use_tib_words = tibetan_words[active_range[0]:active_range[1]+1]


            if TIB or (not TIB and CORR_TIB):
                if correct_tib_loc:
                    overlap = len(range(max(correct_tib_loc, active_range[0]), min(correct_tib_loc + len(accurate_align.split(" "))-1, active_range[1])+1))
                    percent_overlap = overlap/len(accurate_align.split(" "))*100
                    overlap_with_actual[j+1] = percent_overlap
                else:
                    overlap_with_actual[j+1] = "N/A"

                
                

            else:
                overlap_with_actual[j+1] = "N/A"
                



            optimal_index = result_info[rank][0]
            optimal_score = result_info[rank][1]
            
             # insert stars at the end of the found result (and at the beginning)
            use_tib_words.insert(optimal_index-active_range[0]+search_len, "***")
            use_tib_words.insert(optimal_index-active_range[0],'***')

            # form string.
            desc_string = f"{j+1}: Optimal at {optimal_index}, scored {1-optimal_score:.5f}, globally ranked at {rank}. Displaying {active_range[0]} to {active_range[1]}"
            res_string =  " ".join(use_tib_words)
            sub_res_string = []
            
            sub_res_string = [":".join([str(index), f"{1-index_to_cosine[index]:.5f}"]) for index in range(active_range[0], active_range[1]+1) if index in index_to_cosine]
            coda_string = "All subsumed results scores: " + "; ".join(sub_res_string)

            

            text_results.extend([desc_string, res_string, coda_string, "\n"])


        if TOKENIZATION_METHOD == "word":
            search_term = dict_tokenize(search_term, as_string=" ")
        elif TOKENIZATION_METHOD == "term":
            search_term = " ".join(tokenizer.tokenize_string(search_term))
        elif TOKENIZATION_METHOD == "term2":
            search_term = " ".join(tokenizer.tokenize_string(search_term))
        elif TOKENIZATION_METHOD == "char":
            search_term = " ".join(list(search_term))
        
        ratio = len(accurate_align.split(" "))/len(search_term.split(" "))

        correct_tib_loc = get_list_loc(accurate_align.split(" "), tibetan_words)
        if correct_tib_loc:
            all_predicted_ranks.append(tib_result_loc)

        sorted_overlap = sorted(overlap_with_actual.items(), key=lambda x:x[1], reverse=True)
        sorted_overlap = [s for s in sorted_overlap if s[1] != "N/A"]

        if len(sorted_overlap) > 0:
            sorted_overlap = [s for s in sorted_overlap if s[1] > 0.0]
            if len(sorted_overlap) > 0:
                top_5_sort = False
                top_res = False
                full_res = False
                some_top = False
                some_top_5 = False
                for s in sorted_overlap:
                    if s[0] == 1 and s[1] >= FULL_RESULT:    
                        top_res = True
                    if s[0] >= 5 and s[1] >= FULL_RESULT:    
                        top_5_sort = True
                    if s[1] >= FULL_RESULT:
                        full_res = True

                    if s[0] == 1 and s[1] >= PARTIAL_RESULT:
                        some_top = True
                    if s[0] >= 5 and s[1] >= PARTIAL_RESULT:
                        some_top_5 = True

                for s in sorted_overlap:
                    if s[1] >= FULL_RESULT:
                        summary_info["top_full_overlap"].append((i, s[0], s[1]))
                        break

                for s in sorted_overlap:
                    if s[1] >= PARTIAL_RESULT and s[1] < FULL_RESULT:
                        summary_info["top_any_overlap"].append((i, s[0], s[1]))
                        break

                if top_res:
                    summary_info["top_res"] += 1
                if top_5_sort:
                    summary_info["full_overlap_top_5"] += 1
                if full_res:
                    summary_info["full_overlap"] += 1
                if some_top:
                    summary_info["some_top"] += 1
                if some_top_5:
                    summary_info["some_top_5"] += 1
                
                summary_info["some_overlap"] += 1


                sorted_string = "\n".join([f"{s[1]:.2f}% of Tibetan in Result #{s[0]}" for s in sorted_overlap])
            else:
                sorted_string = "Correct match not represented in these results"

        else:
            sorted_string = "Correct Tibetan Unknown"
        if write_output:
            if not os.path.isdir(results_dir):
                os.mkdir(results_dir)
            with open(f'{results_dir}/{i}.txt','w',encoding='utf8') as wf:
                wf.write(f"{search_term}\n")
                wf.write(f"{accurate_align}\n\n")
                wf.write(f"Actual Location of Correct Tibetan: {correct_tib_loc}, Predicted Rank: {tib_result_loc} out of {all_loc_poss};\n\n")
                wf.write(f"{sorted_string}\n\n")
                wf.write(f"Tibetan is {ratio:.2f} times longer, actual adjustment ratio was {used_ratio}\n")
                wf.write(f"avg similarity score: {mean_num:.2f}\nstd of similarity score: {std:.2f}\n\n")
                wf.write("\n".join(text_results))

    clustered_full = []
    clustered_partial = []
    med = "n/a"
    mean = "n/a"
    std = "std"
    if TIB or (not TIB and CORR_TIB):
        mean = statistics.mean(all_predicted_ranks)
        med = statistics.median(all_predicted_ranks)
        std = statistics.pstdev(all_predicted_ranks)

        clustered_full = [s[1] for s in summary_info["top_full_overlap"]]
        clustered_partial = [s[1] for s in summary_info["top_any_overlap"]]
        if len(clustered_full) > 0:
            clustered_full_mean = statistics.mean(clustered_full)
            clustered_full_median = statistics.median(clustered_full)
            missing_full = summary_info['total_phrases'] - len(clustered_full)

            if REPORTING == "verbose":
                print(f"The median clustered full Tibetan appears at rank {clustered_full_median}, average is {clustered_full_mean:.2f}, with {missing_full} phrases with no full results")
            else:
                print(f"Clustered full: {clustered_full_median}, {clustered_full_mean:.2f}")
        if len(clustered_partial) > 0:
            clustered_partial_mean = statistics.mean(clustered_partial)
            clustered_partial_median = statistics.median(clustered_partial)
            missing_partial = summary_info['total_phrases'] - len(clustered_partial)
            if REPORTING == "verbose":
                print(f"The median clustered partial Tibetan appears at rank {clustered_partial_median}, average is {clustered_partial_mean:.2f}, with {missing_partial} phrases with no partial results")
            else:
                print(f"Clustered partial: {clustered_partial_median}, {clustered_partial_mean:.2f}")
        if REPORTING == "verbose":
            print(f"The median predicted global rank for the correct Tibetan is {med} of {all_loc_poss} possibilities\nThe average rank is {mean:.2f}\nStandard deviation is {std:.2f}")
            
            print(f"A total of {summary_info['total_phrases']} Chinese phrases were searched")
            print(f"The full Tibetan was found in the top result {summary_info['top_res']} times.")
            print(f"At least some Tibetan was found in the top result {summary_info['some_top']} times")
            
            print(f"The full Tibetan was found in the top 5 results {summary_info['full_overlap_top_5']} times.")
            print(f"At least some Tibetan was found in the top 5 results {summary_info['some_top_5']} times.")
            
            print(f"The full Tibetan was found in any results {summary_info['full_overlap']} times.")
            print(f"At least some of the Tibetan was found {summary_info['some_overlap']} times.")
        else:
            print(f"Global rank {med}, {mean:.2f}")
    
    if len(clustered_full) > 0:
        return_results = [clustered_full_median, clustered_full_mean, None, None, med, mean]
    else:
        return_results = [None, None, None, None, med, mean]
    if len(clustered_partial) > 0:
        return_results[2] = clustered_partial_median
        return_results[3] = clustered_partial_mean

    #NormAdjust\tShortAdjust\tClusteredFullMedian\tClusteredFullMean\tPartialMedian\tPartialMean\tMedianRes\tMeanRes\tTopResultCount\tFullResultCount\tSomeResultCount\n
    temp_res = [TIB_ADJUST, TIB_ADJUST_SHORT] + return_results[:2]+return_results[4:] + [summary_info['top_res']/summary_info['total_phrases'], summary_info['full_overlap']/summary_info['total_phrases'], summary_info['some_overlap']/summary_info['total_phrases']]
    temp_res = [str(r) for r in temp_res]
    summary_results.append("\t".join(temp_res))

    if write_output:
        with open(f"{results_dir}/results.txt", 'w', encoding="utf8") as wf:
            wf.write(f"Results for {ANALYSIS_FILE}\n\n")
            wf.write(f"The top {TOTAL_RESULTS} results are displayed, clustering when the start of another result is within {CLUSTER_WINDOW} characters\n")
            wf.write(f"Search window is adjusted by a factor of {TIB_ADJUST+1}\n")
            wf.write(f"Tibetan phrases shorter than {SHORT_LENGTH} are adjusted by {TIB_ADJUST_SHORT+1}\n")
            wf.write(f"Chinese embeddings come from {CHIN_EMBED}\nTibetan from {TIB_EMBED}\n")
            
            
            wf.write(f"The distance metric used is {DISTANCE_METRIC}\n\n")

            if TIB or (not TIB and CORR_TIB):
                if len(clustered_full) > 0:
                    wf.write(f"The median clustered full Tibetan appears at rank {clustered_full_median}, average is {clustered_full_mean:.2f}, with {missing_full} phrases with no full results\n")   
                else:
                    wf.write("No full Tibetan results\n")
                    

                if len(clustered_partial) > 0:
                
                    wf.write(f"The median clustered partial Tibetan appears at rank {clustered_partial_median}, average is {clustered_partial_mean:.2f}, with {missing_partial} phrases with no partial results\n")
                else:
                    wf.write("No partial Tibetan results\n")


                wf.write(f"The median predicted Global rank for the correct Tibetan is {med} of {all_loc_poss} possibilities\nThe average rank is {mean:.2f}\nStandard deviation is {std:.2f}\n\n")

                wf.write(f"A total of {summary_info['total_phrases']} Chinese phrases were searched\n")
                wf.write(f"The full Tibetan was found in the top result {summary_info['top_res']} times.\n")
                wf.write(f"At least some Tibetan was found in the top result {summary_info['some_top']} times\n")
                wf.write(f"The full Tibetan was found in the top 5 results {summary_info['full_overlap_top_5']} times.\n")
                wf.write(f"At least some Tibetan was found in the top 5 results {summary_info['some_top_5']} times.\n")
                wf.write(f"The full Tibetan was found in any results {summary_info['full_overlap']} times.\n")
                wf.write(f"At least some of the Tibetan was found in {summary_info['some_overlap']} results.\n")
    return return_results
                

if __name__ == "__main__":
    embeddings = load_embeddings(TIB_EMBED, CHIN_EMBED)

    run(embeddings, True)

    print("finished")
        
    print(len(all_lengths), sum(all_lengths), sum(all_lengths)/len(all_lengths),statistics.median(all_lengths))
    