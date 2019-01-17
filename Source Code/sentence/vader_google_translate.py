from textblob import TextBlob
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import goslate
import time, os
from vaderSentiment import SentimentIntensityAnalyzer

# start = time.time()
# file = []

# def fileRead(folder):
#     for filename in os.listdir(folder):
#         f = open(folder + "/" + filename, encoding="utf-8-sig")
#         for l in f:
#             l.strip()
#             s = l.split("।")
#             file.extend(s)


# # fileRead("accident")
# fileRead("Test")
# print(len(file))
# b = input("Enter Bangla sentence :")

analyzer = SentimentIntensityAnalyzer()
t = analyzer.polarity_scores("If the speed of foreign aid increases and the increase in the revenue collection will reduce the amount of debt.")
print(str(t))
file = open('data.txt', 'r', encoding="utf-8-sig")
file = [i.strip() for i in file]
for line in file:
    line.strip()
    # print(line)
    # textBlobTrans = TextBlob(line)
    # print(textBlobTrans)
    # c = str(textBlobTrans.translate(to='en'))
    # print(c)

    # analyze = TextBlob(c)
    # print('By TextBlob Sentiment analysis')
    # print(analyze.sentiment)

    # print('By vaderSentiment analyzer')
    analyzer = SentimentIntensityAnalyzer()
    # vs = analyzer.polarity_scores(c)

    # print(str(vs))
    #print("By Google:")
    # translator = Translator()
    # s = translator.translate(line, dest="en")
    # print(s.text)
    gs = goslate.Goslate()
    s = gs.translate(line,"en")
    #print(s)
    t = analyzer.polarity_scores(s)
    #print(str(t))

# print(time.time() - start)
# print("By memory:")
# from_lang = "bn"
# to_lang = "en"
# api_url = "http://mymemory.translated.net/api/get?q={}&langpair={}|{}".format(
# 	line, from_lang, to_lang)
# # hdrs = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11','Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3','Accept-Encoding': 'none','Accept-Language': 'en-US,en;q=0.8','Accept-Language': 'en-US,en;q=0.8','Connection': 'keep-alive'}
# hdrs ={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
# 	'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
# 	'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
# 	'Accept-Encoding': 'none',
# 	'Accept-Language': 'en-US,en;q=0.8',
# 	'Connection': 'keep-alive'}
# response = requests.get(api_url, headers=hdrs)
# response_json = json.loads(response.text)
# translation = response_json["responseData"]["translatedText"]
# translator_name = "MemoryNet Translation Service"
# v = analyzer.polarity_scores(translation)
#
# print(str(v))
