# Bengali VADER: A Sentiment Analysis Approach Using Modified VADER
This is a tool for Bengali sentiment analysis, which is built modifying the popular English sentiment analysis tool VADER. This is a lexicon based sentiment analysis tool. It can analyze sentiment of both sentences and documents.

The work is divided into three folders.
* Dictionary
* Related works or paper
* Source code

## Dictionary
The dictionary contains Bengali lexicon that is transaled form English lexicon from VADER. It also contains our created lexicon. The extracted lexicon is created taking review of public using this [site](http://imran03.pythonanywhere.com/). Bengali booster dictionary, negation list and stop-words are also included.

## Instruction of runnig the tool
To run the tool first download python3 from [Python website](https://www.python.org/). Then install [nltk](https://www.nltk.org/) and [numpy](http://www.numpy.org/).
```bash
pip install nltk
```
```bash
pip install numpy
```
In the source code folder the code is given for three different datasets and sentece level analysis.

* To analyze sentiment of the movie review dataset run [sentiment.py](BengaliVADER/Source%20Code/movie%20review/sentiment.py) using bash or command prompt in its file location.
```bash
python sentiment.py
```
* For sports dataset open sports dataset folder and run [sports_sentiment.py](BengaliVADER/Source%20Code/sports%20dataset/sports_sentiment.py).
```bash
python sports_sentiment.py
```
* For twitter dataset open tweet dataset folder and run [tweet_sentiment.py](BengaliVADER/Source%20Code/tweet%20dataset/tweet_sentiment.py).
```bash
python tweet_sentiment.py
```
* To analyze the sentiment of a sentence open the folder sentence and run [sentiment_sentence.py](BengaliVADER/Source%20Code/sentence/sentiment_sentence.py), then enter the sentence you want analyze the sentiment.
```bash
python sentiment_sentence.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
