# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 22:01:21 2024

@author: sreev
"""
# importing the required modules  
from chatterbot import ChatBot  
from chatterbot.trainers import ListTrainer  
# creating a chatbot  
myBot = ChatBot(  
    name = 'Sakura',  
    read_only = True,  
    logic_adapters = [  
        'chatterbot.logic.MathematicalEvaluation',  
        'chatterbot.logic.BestMatch'  
        ]  
        )
# training the chatbot  
small_convo = [  
    'Hi there!',  
    'Hi',  
    'How do you do?',  
    'How are you?',  
    'I\'m cool.',  
    'Always cool.',  
    'I\'m Okay',  
    'Glad to hear that.',  
    'I\'m fine',  
    'I feel awesome',  
    'Excellent, glad to hear that.',  
    'Not so good',  
    'Sorry to hear that.',  
    'What\'s your name?',  
    ' I\'m Sakura. Ask me a math question, please.'  
    ]  
  
math_convo_1 = [  
    'Pythagorean theorem',  
    'a squared plus b squared equals c squared.'  
    ]   
