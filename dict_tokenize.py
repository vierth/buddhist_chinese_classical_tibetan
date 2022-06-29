import re

class Tokenizer:
    def __init__(self, term_file = "karashima_and_dazhidulun.txt"):
        with open(term_file, "r", encoding="utf8") as rf:
            terms = rf.read().split("\n")

        terms = list(set([t.replace("[", "").replace("]", "").replace("(","").replace(")", "") for t in terms if t != ""]))

        sorted_terms = sorted(terms, key=lambda x:len(x), reverse=True)

        self.regex = re.compile("("+"|".join(sorted_terms)+")")

    def tokenize_string(self, input):
        term_list = self.regex.split(input)
        tokenized = []
        for i, t in enumerate(term_list):
            if i % 2 == 0:
                if t != "":
                    tokenized.extend(list(t))
            else:
                tokenized.append(t)

        return tokenized