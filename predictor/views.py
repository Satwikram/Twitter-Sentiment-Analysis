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

        # Cleaning the tweet
        tweet = clean_tweet(tweet)

        # Loading Tokenizer
        encoder = tfds.features.text.SubwordTextEncoder.load_from_file("Tokenizer/tokenizer")
        tweet = np.array([tweet])
        print("Tweet is:", tweet)
        print("Tweet of numpy is:",tweet[0])

        # Encoding the tweet
        tweet = encoder.encode(tweet[0])
        print("Encoded text:",tweet)

        # Loading the saved model
        print("Loading model...")
        model = tf.saved_model.load("sentiment\saved_model.pb")

        print("Signatures of model:",print(list(model.signatures.keys())))

        predictor = model.signatures["serving_default"]
        print(predictor.structured_outputs)

        result = predictor(tf.constant(tweet))

        result['output_1'][0][0]

        if result['output_1'][0][0] < .5:
            message = "Oops!, thats a negative twwet :("

        else:
            message = "Its a positive tweet :)"

        return render(request, 'results.html', {'message': message})
    else:
        return render(request, 'predict.html')