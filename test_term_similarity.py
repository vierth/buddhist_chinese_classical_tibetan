import random

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dict_tokenize import Tokenizer
from embedding_pipeline import load_embeddings, load_single_embeds
from pavdhutils.tokenize1 import dict_tokenize
from pavdhutils.vector_analysis import average_vectors, compare_vectors

show_english = True
centered = False
WORD_OR_CHAR = "term2"
# def vectorize_phrase(phrase, embeddings, word_char):
#     if word_char != "char":
#         if phrase in embeddings:
#             return embeddings[phrase]

#     else:
#         if all(w in embeddings for w in phrase):
#             return average_vectors([embeddings[w] for w in phrase if w in embeddings])


def vectorize_phrase(phrase, embeddings, word_or_char="char"):
    if phrase in embeddings:
        return embeddings[phrase]
    if word_or_char == "char":
        return average_vectors([embeddings[w] for w in phrase if w in embeddings])
    elif word_or_char == "word":
        phrase = dict_tokenize(phrase, as_string=" ")
        print(phrase)
        return average_vectors([embeddings[w] for w in phrase.split(" ") if w in embeddings])
    elif word_or_char == "term" or word_or_char == "term2":

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


if WORD_OR_CHAR == "word":
    tib_embed = 'tib_comb_word.p'
    chin_embed = 'chin_comb_word.p'
elif WORD_OR_CHAR == "char":
    tib_embed = 'tib_comb.p'
    chin_embed = 'chin_comb.p'
elif WORD_OR_CHAR == "term":
    tib_embed = 'tib_comb_hybrid_term.p'
    chin_embed = 'chin_comb_hybrid_term.p'
    tokenizer = Tokenizer("karashima_terms.txt")
elif WORD_OR_CHAR == "term2":
    tib_embed = 'tib_comb_hybrid_term_2.p'
    chin_embed = 'chin_comb_hybrid_term_2.p'
    tokenizer = Tokenizer("karashima_and_dazhidulun.txt")

# embeddings = load_single_embeds("chinese_model_hybrid_char_term_2.vec")
# print(just_chin)

embeddings, _, _ = load_embeddings(tib_embed, chin_embed, center=centered)
all_res = []
random_res = []
used_embeds_labels = []
used_embeds = []
paired_term = []
relationship = []
lang = []
term_cats = []
eng = []
# use_words = ["有力", "無力能", "所苦", "有樂", "智人"]
# use_words = ["菩薩", "縁", "不善", "善", "好", "惡", "壞"]
# use_words = ["大王"]

english_dict = {
    "狗": "dog",
    "犬": "dog",
    "貓": "cat",
    "熊": "bear",
    "鵝": "goose",
    "鳥": "bird",
    "蛇": "snake",
    "豹": "leopard",
    "蝦": "prawn/shrimp",
    "蚵": "oyster",
    "蝎": "scorpion",
    "蟲": "insect/animals",
    "馬": "horse",
    "牛": "cow/buffalo",
    "螞": "ant",
    "蟻": "ant",
    "蜂": "bee",
    "虎": "tiger",
    "蚊": "mosquito",
    "獸": "beast/animal",
    "禽": "beast/animal",
    "雞": "chicken/fowl",
    "鵲": "magpie",
    "鷄": "chicken",
    "羊": "sheep",
    "驢": "donkey",
    "象": "elephant",
    "魚": "fish",
    "蛙": "frog",
    "鴿": "pigeon",
    "སྲོག་ཆགས": "creature/vermin",
    "སྦྲུལ": "snake",
    "སྟག": "tiger",
    "གཟིག": "leopard",
    "དོམ": "bear",
    "ཕྲུ་གུ": "child",
    "བོང་བུ": "donkey",
    "ཐི་བ": "pigeon",
    "ངང་པ": "goose",
    "རྟ": "horse",
    "གླང་པོ": "elephant",
    "གྲོག་སྦུར": "ant",
    "སྦྲང་མ": "bee",
    "སྦྲང་བུ": "bee/fly",
    "梵宮": "palaces of the Brahmā",
    "梵音": "Brahmā’s voice",
    "梵王": "Brahma King",
    "梵行": "Brahma-conduct",
    "梵志": "Brahman",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
    "百": "100",
    "千": "1000",
    "萬": "10000",
    "東": "east",
    "南": "south",
    "西": "west",
    "北": "north",
    "春": "spring",
    "夏": "summer",
    "秋": "autumn",
    "冬": "winter"
}

not_people = ["非人", "鬼", "神", "龍"]
# brahman = ["梵宮","梵音","梵王","梵行","梵志"]
animals = "狗犬貓熊鵝鳥蛇豹蝦蚵蝎蟲馬牛螞蟻蜂虎蚊獸禽雞鵲羊驢象魚蛙鴿"
numbers = "一二三四五六七八九十百千萬"
directions = "東南西北"
seasons = "春夏秋冬"


def categorize_term(term):
    # if term in not_people:
    #     return "not person"
    if term in numbers:
        return "number"
    elif term in directions:
        return "direction"
    elif term in seasons:
        return "season"
    elif term in animals:
        return "animal"
    # elif term in brahman:
    #     return "brahamn"
    # elif "菩薩" in term:
    #     return "boddhisatva"
    # elif "魔" in term:
    #     return "demon"
    else:
        return "other"


with open('tibfreqwords.txt', 'r', encoding='utf8') as rf:
    text = rf.read()
    terms = text.split("\n")
    terms = terms[:100]

use_words = terms
with open('wordpairs.txt', 'r', encoding='utf8') as rf:
    text = rf.read().split("\n")
    data = [d.split(",") for d in text if d != ""]
    results = []
    for d in data:
        if d[1] in terms:
            use_words.append(d[0])
        chin = vectorize_phrase(d[0], embeddings, WORD_OR_CHAR)
        if d[0] in embeddings and d[0] not in used_embeds_labels:

            chin = embeddings[d[0]]
            # if all(c in embeddings for c in d[0]) and d[1] in embeddings:
            if (type(chin) == np.ndarray or type(chin) == list) and d[1] in embeddings:
                # chin = vectorize_phrase(d[0], embeddings)
                # chin = embeddings[d[0]]

                tib = embeddings[d[1]]
                # res = 1 - distance.cosine(chin, tib)

                # all_res.append(res)
                # results.append([d[0], d[1], res])

                # embed terms
                if d[0] not in used_embeds_labels:
                    used_embeds.append(chin)
                    used_embeds_labels.append(d[0])
                    paired_term.append(d[1])
                    relationship.append(f"{d[0]},{d[1]}")
                    lang.append('c')
                    term_cats.append(categorize_term(d[0]))
                    if d[0] in english_dict:
                        eng.append(d[0]+f"{english_dict[d[0]]}")
                    else:
                        eng.append("")
                if d[1] not in used_embeds_labels:
                    used_embeds.append(tib)
                    used_embeds_labels.append(d[1])
                    paired_term.append(d[0])
                    relationship.append(f"{d[0]},{d[1]}")
                    lang.append('t')
                    term_cats.append(categorize_term(d[0]))
                    if d[0] in english_dict:
                        if d[1] in english_dict:
                            eng.append(d[1]+f"{english_dict[d[1]]}")
                        else:
                            eng.append(d[1]+f"{english_dict[d[0]]}")
                    else:
                        eng.append("")

                # rand_term = data[random.randint(0,len(data)-1)]
                # while rand_term[1] not in embeddings:
                #     rand_term = data[random.randint(0,len(data)-1)]

                # random_res.append(1-distance.cosine(chin, embeddings[rand_term[1]]))

# sorted_res = sorted(results, key=lambda x:x[2], reverse=True)

# sorted_res = [",".join([r[0], r[1], str(r[2])[:5]]) for r in sorted_res]
# mean = np.mean(all_res)
# median = np.median(all_res)
# std = np.std(all_res)
# max_val = np.max(all_res)
# min_val = np.min(all_res)

# print(use_words)
# rand_mean = np.mean(random_res)
# rand_median = np.median(random_res)
# rand_std = np.std(random_res)
# rand_max_val = np.max(random_res)
# rand_min_val = np.min(random_res)

# with open(f'word_similarities_{centered}_{WORD_OR_CHAR}.txt', 'w', encoding='utf8') as wf:
#     print(f"mean: {mean:.2f}, median: {median:.2f}, std: {std:.2f}, max: {max_val:.2f}, min: {min_val:.2f}\n")
#     # print(f"randomized matches: {rand_mean:.2f}, median: {rand_median:.2f}, std: {rand_std:.2f}, max: {rand_max_val:.2f}, min: {rand_min_val:.2f}\n")
#     wf.write(f"mean: {mean:.2f}, median: {median:.2f}, std: {std:.2f}, max: {max_val:.2f}, min: {min_val:.2f}\n")
#     # wf.write(f"randomized matches: {rand_mean:.2f}, median: {rand_median:.2f}, std: {rand_std:.2f}, max: {rand_max_val:.2f}, min: {rand_min_val:.2f}\n")
#     wf.write("\n".join(sorted_res))
# pca = PCA(n_components=50)
# reduced_data = pca.fit_transform(used_embeds)

tsne = TSNE(random_state=42)  # 6 isn't bad
reduced_data = tsne.fit_transform(used_embeds)


dim_1 = [d[0] for d in reduced_data]
dim_2 = [d[1] for d in reduced_data]
# lines = [[[d[0], d[1]],[reduced_data[i+1][0],reduced_data[i+1][1]]] for i,d in enumerate(reduced_data[::2])]
# df = pd.DataFrame({"term":used_embeds_labels, "Dim1":dim_1, "Dim2":dim_2, "language":lang, "pairedterm":paired_term, "rel":relationship})
# for l, p,c in zip(used_embeds_labels,relationship,term_cats):
#     if c == "animal":
#         print(l,p)
df = pd.DataFrame({"term": used_embeds_labels, "eng": eng, "Dim1": dim_1, "Dim2": dim_2,
                  "Category": term_cats, "language": lang, "pairedterm": paired_term, "rel": relationship})
print(df[df.Category == "animal"])
df = df[df.Category != "other"]

if show_english:
    text_val = "eng"
else:
    text_val = "term"


fig = px.scatter(data_frame=df, x="Dim1", y="Dim2", color="Category",
                 text=text_val, hover_data=["term", "rel", "pairedterm"])
fig.for_each_trace(lambda t: t.update(
    textfont_color=t.marker.color, textfont_size=10, marker_opacity=0))
fig.update_layout(yaxis_range=[-42, -32.5])
fig.update_layout(xaxis_range=[16, 29.5])
fig.update_layout(showlegend=False)
fig.write_image("detail.pdf")
# fig.write_html('all_embeds_fig.html')
fig.show()

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import plotly.graph_objects as go
# from dash.dependencies import Input, Output, State
# #https://community.plotly.com/t/highlight-group-of-points-on-hover/44278/3
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),

#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),

#     dcc.Graph(
#         id='graph',
#     )
# ])
# df = df[:1000]
# # DEFAULT_SIZE = 10
# # HIGHLIGHT_SIZE = 30
# DEFAULT_COLOR = "gray"
# HIGHLIGHT_COLOR = "blue"
# # styles = {r:dict(marker=dict(size=DEFAULT_SIZE)) for r in relationship}
# styles = {r:dict(marker=dict(color=DEFAULT_COLOR)) for r in relationship}

# SELECTED = None

# @app.callback(
#     Output('graph','figure'),
#     Input('graph', 'hoverData'),
#     State('graph', 'figure'))

# def update_graph(hover_data, fig):
#     global SELECTED
#     if hover_data is not None:
#         if SELECTED is not None:
#             # fig['data'][SELECTED]['marker']['size'] = DEFAULT_SIZE
#             fig['data'][SELECTED]['marker']['color'] = DEFAULT_COLOR
#         rel = hover_data['points'][0]['customdata']
#         SELECTED = ix = [scatter['name'] for scatter in fig['data']].index(rel)
#         # fig['data'][ix]['marker']['size'] = HIGHLIGHT_SIZE
#         fig['data'][ix]['marker']['color'] = HIGHLIGHT_COLOR
#     else:
#         fig = go.Figure()
#         for rel, terms in df.groupby('rel'):
#             fig.add_scatter(
#                 x=terms.Dim1,
#                 y=terms.Dim2,
#                 name=rel,
#                 customdata=terms.rel,
#                 mode='markers',
#                 **styles[rel]
#             )
#         fig.update_layout(height=1000, hovermode='closest', uirevision='static')
#     #fig = px.scatter(data_frame=df, x="Dim1", y="Dim2", color="language", hover_data=["term","pairedterm"])
#     return fig

# if __name__ == '__main__':
#     app.run_server(debug=True)
