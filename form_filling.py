import pandas as pd
import spacy

from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode
from collections import OrderedDict
from spacy.lang.fr.stop_words import STOP_WORDS as stopwords

pd.options.mode.chained_assignment = None

PATH = "enregistrement/"

with open(PATH+"tmp_error_not_found.txt", "w") as f:
    pass
with open(PATH+"tmp_error_found.txt", "w") as f:
    pass
with open(PATH+"tmp_multiples_errors_code.txt", "w")as f:
    pass

def create_dico_acro():
    data = pd.read_excel(PATH+"BCD_BO_JC_ABREV_ELT_INC.xlsx")
    data = data.fillna("")
    dico = {}
    for _, row in data.iterrows():
        dico[row['ABR'].lower()] = row["LIB"]
    return dico

def initial_prompt():
    with open(PATH+"initial_prompt.txt", "r") as f:
        return f.readline()

def no_punct(word):
    for letter in word:
        if letter in ".?!:',()\"":
            return False
    return True

def stay_united(words):
    while "-" in words:
        index = words.index("-")
        if 1 <= index <= len(words) - 2:
            words = words[:index-1] + [words[index-1] + "-" + words[index+1]] + words[index+2:]
    return words

def stay_separate(words):
    texte = " ".join(words)
    return texte.split(" ")

def unacronym(word, dico_abr):
    if word.lower() in dico_abr:
        return dico_abr[word.lower()]
    return word

def words_cleaner(text, dico_abr, stopwords, nlp, stemmer):
    removal_list = ["son", "avant", "après", "bas"]
    for word_remove in removal_list:
        if word_remove in stopwords:
            stopwords.remove(word_remove)
    return [unidecode(stemmer.stem(word.lower())) for word in
            stay_separate(stay_united([unacronym(i.text, dico_abr) for i in nlp(text)]))
            if no_punct(word) and word.lower() not in stopwords]

def create_graph(df):
    knowledge_graph = {}
    for index, row in df.iterrows():
        words = list(OrderedDict.fromkeys(row["elt_inc"].split(" ")))
        for word in words:
            if word in knowledge_graph:
                knowledge_graph[word] += [words]
            else:
                knowledge_graph[word] = [words]
    return knowledge_graph

def find_relevant_default(stem_words, knowledge_graph):
    scores = {i: 0 for i in knowledge_graph.keys()}
    for stem_word in stem_words:
        if stem_word in knowledge_graph:
            scores[stem_word] += 1
    return {i: knowledge_graph[i] for i in scores.keys() if scores[i] >= 1}

def best_matching_score(stem_words, targets):
    max_score = 0
    max_index = []
    max_key = []
    for key in targets:
        for i in range(len(targets[key])):
            list_elt_inc = targets[key][i]
            number_match = 0
            for stem_word in stem_words:
                if stem_word in list_elt_inc:
                    number_match += 1
            score = number_match/len(list_elt_inc)
            if score > max_score:
                max_score = score
                max_index = [i]
                max_key = [key]
            elif score == max_score:
                max_index += [i]
                max_key += [key]
    if max_score < 0.4:
        return [], [], 0
    return max_key, max_index, max_score

def find_answer(df, max_key, max_index, max_score, knowledge_graph):
    i = -1
    list_unique = []
    answer = []
    for key in max_key:
        i += 1
        if knowledge_graph[key][max_index[i]] in list_unique:
            continue
        list_unique += [knowledge_graph[key][max_index[i]]]
        answer += [" ".join(knowledge_graph[key][max_index[i]])]
    result = df[df.elt_inc_stem_unique.isin(answer)]
    result["confiance"] = max_score
    result["frequence"] = result["count"]/sum(result["count"])
    if max_score == 1:
        result["longueur"] = result.apply(lambda x: len(x["elt_inc"].split(" ")), axis=1)
        return result[
            ['elt_id', 'elt', 'inc_id', 'inc', 'frequence', 'confiance']
        ].loc[result["longueur"].idxmax()]
    if max(result["frequence"]) > 0.5:
        return result[
            ['elt_id', 'elt', 'inc_id', 'inc', 'frequence', 'confiance']
        ].loc[result["frequence"].idxmax()]
    return result[['elt_id', 'elt', 'inc_id', 'inc', 'frequence', 'confiance']].head(5)

def find_error_code(df, stem_words, knowledge_graph):
    for inc_lvl in ["v1", "v2", "v3", "1", "2", "3"]:
        if inc_lvl in stem_words:
            break
    else:
        inc_lvl = "Not found"
    inc_lvl = {"v1": "v1",
               "v2": "v2",
               "v3": "v3",
               "1": "v1",
               "2": "v2",
               "3": "v3",
               "Not found": "Not found"}[inc_lvl]
    targets = find_relevant_default(stem_words, knowledge_graph)
    if targets == {}:
        return "No match found for this error"
    max_key, max_index, max_score = best_matching_score(stem_words, targets)
    if max_score == 0:
        return "No match found for this error"
    answer = find_answer(df, max_key, max_index, max_score, knowledge_graph)
    answer["inc_lvl"] = inc_lvl
    return answer


nlp = spacy.load("fr_core_news_lg")
print("NLP model loaded")
stemmer = SnowballStemmer(language='french')
print("stemmer model loaded")
df = pd.read_csv(PATH+"BO_liste_defaut.csv")
print("Operator manuel loaded")
prompt = initial_prompt()
print("Keyword loaded")
dico_abr = create_dico_acro()
print("Acronym translation loaded")
knowledge_graph = create_graph(df)
print("Knowledge graph loaded")

def to_file(text, df=df, knowledge_graph=knowledge_graph, dico_abr=dico_abr,
            stopwords=stopwords, nlp=nlp, stemmer=stemmer):
    stem_words = list(OrderedDict.fromkeys(words_cleaner(text, dico_abr, stopwords, nlp, stemmer)))
    answer = find_error_code(df, stem_words, knowledge_graph)
    print(answer)
    if type(answer) == str:
        with open(PATH+"tmp_error_not_found.txt", "a") as f:
            f.write(text+"\n")
    elif len(answer) == 7:
        with open(PATH+"tmp_error_found.txt", "a") as f:
            f.write(text+"\n")
            f.write(str(answer)+"\n")
    elif len(answer) <= 5:
        with open(PATH+"tmp_multiples_errors_code.txt", "a")as f:
            f.write(text+"\n")
            f.write(str(answer)+"\n")
    return 0
