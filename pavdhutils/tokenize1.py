
from pavdhutils.cleaning import clean
from pavdhutils.general import load_text
import stanza,re, itertools
stanza.download('lzh')
nlp = stanza.Pipeline('lzh')
# pos_nlp = stanza.Pipeline('lzh',processors='tokenize,pos', tokenize_pretokenized='True')


# split the text
def graph_tokenizer(text):
    return text.split("\n\n")

def sent_tokenizer(text):
    return re.split(r'[\n。？！\．]', text)

def phrase_tokenizer(text):
    return re.split(r'[。？！；：；、，「」『』《》 *  ()○]', text)

def flatten(input_list):
    new_list = list(itertools.chain(*input_list))
    return [l for l in new_list if l != ""]

def get_dict_regex(filepath):
    with open(filepath, 'r', encoding='utf8') as rf:
        vocab = sorted(list(set(rf.read().split("\n"))),key=len,reverse=True)
    return re.compile("("+("|").join(vocab)+")"), set(vocab)

def dict_tokenize(text, dict_location="buddhistdict.txt", as_string=False, get_pos=False):
    term_regex,vocab = get_dict_regex(dict_location)
    paragraphs = graph_tokenizer(text)

    # Clean the paragraphs and then find the dictionary words, marking up with <w>
    # tags
    paragraphs = [term_regex.sub(r'<w>\1</w>', p) for p in paragraphs]

    # Break into sentences. Another step that is not super necessary, but results
    # in a nice output (sentences divided into words)
    sentences = [sent_tokenizer(p) for p in paragraphs]

    # flatten the list from a list of paragraphs, each of which is a list of sentences
    # to just a list of sentences
    sentences = flatten(sentences)

    # remove any sentence that has no words in it
    sentences = [s for s in sentences if s != ""]

    # create an empty list to contian the parsed output
    parsed = []
    # iterate through each sentence and parse it as necessary
    for sentence in sentences:

        # keep track of words in the current sentence
        temp_parse = []

        # futher reduce the text into phrases, as words will not cross phrase boundaries
        for phrase in phrase_tokenizer(sentence):

            # keep track of the words in the current phrase
            temp_words = []
            
            # find the first marked up word
            word_start = phrase.find("<w>")

            # as long as there are still marked up words, keep searching
            while word_start != -1:
                # if an identified word is at the beginning of the paragraph then append it to
                # the parsed list
                if word_start == 0:
                    word_end = phrase.find("</w>")
                    temp_words.append(phrase[word_start+3:word_end])
                    # move the text forward past the marked-up word
                    phrase =phrase[word_end+4:]
                else:
                    # specify the chunk up to the next identified word
                    chunk = phrase[:word_start]
                    # if the chunk is just one character, then append it to the list
                    if len(chunk) == 1:
                        temp_words.append(chunk)
                    # otherwise, use the alternative parser method and EXTEND the list 
                    # assuming the alternative parser returns a list of words
                    else:
                        temp_words.extend(list(chunk))
                        # alternatively you could switch this for:
                        #temp_words.extend([w.text for w in nlp(chunk).sentences[0].words])
                        
                    # move the text up to the next identified word
                    phrase = phrase[word_start:]
                # search for the next identified word. Once there are no more words,
                # you will exit out of the while loop.
                word_start = phrase.find("<w>")

            # there is a possiblity that there are no more marked up words, but still 
            # untokenized text!

            # check if there is remaining text
            remaining = len(phrase)

            # if there is remaining text, then parse it
            if remaining > 0:
                # if it is just one character save it
                if remaining == 1:
                    temp_words.append(phrase)
                # otherwise, use a parser to extend the list
                else:
                    # temp_words.extend(list(phrase))
                    # alternatively you could switch this for:
                    temp_words.extend([w.text for w in nlp(phrase).sentences[0].words])
            temp_parse.extend(temp_words)
        
        # as long as there are words in the temp_parse list, then append it to the
        # parsed list
        if len(temp_parse) > 0:
            parsed.append(temp_parse) 
    if as_string:
        parsed = as_string.join([" ".join(p) for p in parsed])
    return parsed

def stanza_tokenize(text, as_string=False):
    doc = nlp(text)
    
    parsed = []
    for sentence in doc.sentences:
        word_list = ["_".join([word.text, word.pos]) for word in sentence.words]
        parsed.append(word_list)

    if as_string:
        parsed = "\n".join([" ".join(s) for s in parsed])
        

    return parsed

def dict_alt(text, as_string=False):


    term_regex,vocab = get_dict_regex(dict_location)
    sentences = sent_tokenizer(text)
    all_sents_fixed = []
    for sent in sentences:
        tokenized_sent = []

if __name__ == "__main__":
    text = load_text("T310(27).txt")
    text = clean(text, preserve_punc=True, preserve_double_graph=True, delwhitespace=False, delnonnewline=True)
    print(text[:1000])
    dict_tokenize(text)
    tokenized = dict_tokenize(text, as_string=True)
    # tokenized = stanza_tokenize(text, as_string=True)
    
    with open('t310_dict_pos.txt','w', encoding='utf8') as wf:
        wf.write(tokenized)