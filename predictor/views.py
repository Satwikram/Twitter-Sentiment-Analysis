from django.shortcuts import render
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import re
from bs4 import BeautifulSoup


def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet,"lxml").get_text()
    tweet = re.sub(r"@[A-Za-z0-9]+", "", tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9./]", "", tweet)
    tweet = re.sub(r"[^A-Za-z]", "", tweet)
    tweet = re.sub(r" +", "", tweet)
    return tweet

def home(request):
    return render(request, 'predict.html')

def predict(request):
    if request.method == "POST":
        tweet = request.POST['tweet']
        tweet = clean_tweet(tweet)

        encoder = tfds.features.text.SubwordTextEncoder.load_from_file("Tokenizer/tokenizer")
        tweet = np.array([tweet])
        print("Tweet is:", tweet)
        print("Tweet of numpy is:",tweet[0])

        tweet = encoder.encode(tweet[0])
        print("Encoded text:",tweet)




        return render(request, 'results.html')
    else:
        return render(request, 'predict.html')