#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 22:33:47 2022

@author: priteshsrivastava
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
# Import the required module for text 
# to speech conversion
from gtts import gTTS
  
# This module is imported so that we can 
# play the converted audio


import speech_recognition as sr
import pyttsx3
 
# Initialize the recognizer

import os  
# Language in which you want to convert
language = 'en'

dontknow = ["I don't know everything, try Dr. Google perhaps", "Easy there SpiderMan, I don't know everything", "Try contacting Pritesh for this", "I know nothing John Snow"]
with open("json_file/intents.json") as file:
    data = json.load(file)

print(data['intents'])

try:
    with open("data.pickle",'rb') as f:
        words, labels, training, output = pickle.load(f)
    
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
            
            if intent['tag'] not in labels:
                labels.append(intent['tag'])
                
                
    words = [stemmer.stem(w.lower()) for w in words if w!= "?"]   
    words = sorted(list(set(words)))      
    labels = sorted(labels)   
    
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
        
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle",'wb') as f:
        pickle.dump((words, labels, training, output ), f)

tf.reset_default_graph()
net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = 'softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    x
    model.load('chat.tflearn')
except:
    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
    model.save('chat.tflearn')
    
def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words= [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)



def listen_jarvis():
    r = sr.Recognizer()
    try:
         
        # use the microphone as source for input.
        with sr.Microphone() as source2:
             
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("I'am ready")
            #listens for the user's input
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            print(MyText)
            return MyText
#            SpeakText(MyText)
             
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        z = 'Samajh nhi aa rha'
        return z
         
    except sr.UnknownValueError:
        print("unknown error occured")
        z = 'kuch Samajh nhi aa rha'
        return z

def chatting():
    print("start talking : ")
    while (1):
        
#        inp = input("You : ")
        inp = listen_jarvis()
        print('came out of function')
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] >0.65:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            mytext = random.choice(responses)
            print(mytext)
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save("welcome.mp3")
  
            # Playing the converted file
            os.system("mpg321 welcome.mp3")  
            
        else:
            
            mytext = random.choice(dontknow)
            print(mytext)
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save("welcome.mp3")
  
            # Playing the converted file
            os.system("mpg321 welcome.mp3")  
            

chatting()    



 
# Function to convert text to
# speech
#def SpeakText(command):
#     
#    # Initialize the engine
#    engine = pyttsx3.init()
#    engine.say(command)
#    engine.runAndWait()
     
     
# Loop infinitely for user to
# speak
  
     
    # Exception handling to handle
    # exceptions at the runtime
