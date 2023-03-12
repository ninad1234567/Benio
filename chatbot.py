import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow
import keras
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('Benio.h5')

#to clean the sentence

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


#bag of wrds

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag =[0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


#to check the cetainity of a perticular ans
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]),verbose = 0)[0]
    ERROR_THRESHOLD = 0.85




    #trial


   #results =[[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results = [[i, r] if r > ERROR_THRESHOLD else [i, 0] for i, r in enumerate(res)]
    print(results)

    if results ==[0,0]:
        return 0
        print("sorry cannot function")
    else:


              #trail
        results.sort(key=lambda x: x[1], reverse=True)
        return_list =[]
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
            print(return_list)
        return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Benio is running!!!!!")


while True:
    message = input(">>>")
    ints = predict_class(message)
    res = get_response(ints,intents)
    print(res)