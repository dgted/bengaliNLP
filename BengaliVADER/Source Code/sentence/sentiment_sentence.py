import math,re,os
import numpy as np
from nltk.util import ngrams

st = open('stopwords.txt', 'r', encoding="utf-8-sig")
PUNC_LIST = ["।", "!", "?", ",", ";", "ঃ", "\"", "-", "(", ")", "[", "]"]
inc = .293
dec = -.293
booster_dic = {"অতি": inc, "অতিশয়": inc, "বেশি": inc, "অনেক": inc, "কম": dec,
               "অল্প": dec, "অধিক": inc, "অধিকতর": inc, "বহুত": inc, "খুব": inc, "সবচেয়ে": inc}

negation_list = ['না', 'নি', 'নয়', 'নাই', 'নেই','নহে']
s = [i.strip() for i in st]
Precision = []
Recall = []
Accuracy = []
f1_score = []

def cross_fold():
    
    dataset = open('data.txt', 'r', encoding='utf-8-sig')
    dataset = [i.strip() for i in dataset]
    
    
    for line in dataset:
        # lines = content.split("।")
        #valence_list = []
        # for line in lines:
        line.strip()
        token_list = remove_punc(line)
        token_list = remove_stopwords(token_list)
        #print(token_list)
        token_line = []
        token_stem = []
        #print(token_line)
        for word in token_list:
            
            if token_list.index(word) == len(token_list)-1:
                stem_word, last = stem_verb(word)
                token_stem.append(stem_word)
                if last != "":
                    token_stem.append(last)
            else:
                stem_word = _stem_verb_step_1(word)
                if stem_word in lex_dic:
                    token_stem.append(stem_word)
                else:
                    stem_word = _stem_verb_step_2(word)
                    if stem_word in lex_dic:
                        token_stem.append(stem_word)
                    else:
                        token_stem.append(word)

        #print(token_stem)
        for word in token_stem:
            if word != "":
                token_line.append(word)
        #print(token_line)
        # Bi-gram word
        bigrams_list = ngrams(token_line,2)
        bigram_token_list = []
        remove_token = []
        for bigram in bigrams_list:
            bigram_words = bigram[0]+"_"+bigram[1]
            #bigram_words = "_".join(list(bigram))
            if bigram_words in lex_dic:
                bigram_token_list.append(bigram_words)
                remove_token.append(bigram[0])
                remove_token.append(bigram[1])
        
        # Tri-gram word
        trigrams_list = ngrams(token_line,3)
        trigram_token_list = []
        for trigram in trigrams_list:
            trigram_words = trigram[0]+"_"+trigram[1]+"_"+trigram[2]
            #bigram_words = "_".join(list(bigram))
            if trigram_words in lex_dic:
                trigram_token_list.append(trigram_words)
                remove_token.append(trigram[0])
                remove_token.append(trigram[1])
                remove_token.append(trigram[2])
        for w in remove_token:
            ind = token_line.index(w)
            token_line.pop(ind)
            
        token_line = trigram_token_list + token_line
        token_line = bigram_token_list + token_line

        # print(token_line)
        sentiments = []
        for item in token_line:
            valence = 0
            i = token_line.index(item)
            if item in booster_dic:
                sentiments.append(valence)
                continue
            sentiments = words_valence(valence, token_line, item, i, sentiments)
            # print(sentiments)
         
        valence_content = score_valence(sentiments)
    
    
        #print(valence_list)
        #valence_content = np.mean(valence_list)
        print(line, valence_content)

        
        
        

lex_dic = dict()
# reading lexicon and word's polarity
def read_lexicon(fileName):
	
	lex = open(fileName, 'r', encoding="utf-8-sig")
	for l in lex:
	    l.strip()
	    # splitting the line on tab and storing
	    # key and value to different variables
	    (w, v) = l.strip().split('\t')[0:2]
	    lex_dic[w] = float(v)
	#print(lex_dic)


# removing punctuation from data
def remove_punc(line):
    token_line = []
    l = filter(None, re.split("[,(,), \-!?:\d+\\n]+", line))
    for token in l:
        if token not in PUNC_LIST:
            token_line.append(token)
    return token_line


# removing stopwords from data
def remove_stopwords(token):
    token_af_rem_st = []
    for t in token:
        if t not in s:
            token_af_rem_st.append(t)
    return token_af_rem_st


# checking for booster words
def booster_check(valence, word):
    scalar = 0.0
    if word in booster_dic:
        scalar = booster_dic[word]
        if valence < 0:
            scalar *= -1
    return scalar

def negation_check( valence, word,intensifier):

    if word in negation_list and intensifier == 1:
        valence *= 0.5
    elif word in negation_list and intensifier == 0:
        valence *= -1

    return valence

# calculating valence of the sentence
def words_valence(valence, token_line, item, i, sentiments):
    if item in lex_dic:
        valence = lex_dic[item]
        intensifier = 0
        
        for s_i in range(0, 3):
            if i > s_i and token_line[i - (s_i + 1)] not in lex_dic:
                s = booster_check(valence, token_line[i - (s_i + 1)])
                if s:
                    intensifier = 1
                if s_i == 1:
                    s *= 0.9
                elif s_i == 2:
                    s *= 0.75
                valence = valence + s
        # negation handle
        for s_i in range(0, len(token_line)-i-1):
            if token_line[i + s_i + 1] not in lex_dic:
                valence = negation_check(valence, token_line[i+s_i+1],intensifier)

    sentiments.append(valence)
    return sentiments

def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score/math.sqrt((score*score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score

def separate_sentiment_scores(sentiments):
    # want separate positive versus negative sentiment scores
    pos_sum = 0.0
    neg_sum = 0.0
    neu_count = 0
    for sentiment_score in sentiments:
        if sentiment_score > 0:
            pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
        if sentiment_score < 0:
            neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
        if sentiment_score == 0:
            neu_count += 1
    return pos_sum, neg_sum, neu_count

def score_valence(sentiments):
    compound = 0
    if sentiments:
        sum_s = float(sum(sentiments))
        # print(sum_s)
        compound = normalize(sum_s)
        # print(compound)

        # discriminate between positive, negative and neutral sentiment scores
        pos_sum, neg_sum, neu_count = separate_sentiment_scores(sentiments)
        #print(pos_sum,neg_sum,neu_count)

        total = pos_sum + math.fabs(neg_sum) + neu_count
        pos = math.fabs(pos_sum / total)
        neg = math.fabs(neg_sum / total)
        neu = math.fabs(neu_count / total)

        # sentiment_dict ={"neg": round(neg, 3),
        #      "neu": round(neu, 3),
        #      "pos": round(pos, 3),
        #      "compound": round(compound, 4)}

    return compound
def _stem_verb_step(word: str) -> str:
    if word.endswith(('না')):
        return word[:-2],'না'
    elif word.endswith(('নি')):
        return word[:-2],'নি'
    return word,''

def _stem_verb_step_1(word: str) -> str:
    if word.endswith(('া', 'ে', 'ি', 'ো', 'ী','য়')):
        return word[:-1]
    return word
def _stem_verb_step_2(word: str) -> str:
    if word.endswith(('ের')):
        return word[:-2]
    return word


def stem_verb(word: str) -> str:
    stemmed, last = _stem_verb_step(word)
    return stemmed, last

read_lexicon('lexicon.txt')
read_lexicon('extracted_lexicon.txt')
print(len(lex_dic))

cross_fold()
    
