from chatterbot import Chatbot
from chatterbot.trainers import ChatterbotCorpusTrainer

#import twitchio
#import tweepy
#import discord.py

# Needs to have Twitter, Twitch and Discord functionality 




saltbot = ChatBot(
    'saltbot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite'
)
trainer = ChatterBotCorpusTrainer(saltbot)
trainer.train(
    'chatterbot.corpus.english' #going to put chat/discord training data here
)

while True:
    try:
        user_input = input('You: ')
        bot_response = saltbot.get_response(user_input)
        print('Bot:', bot_response)
        
    except(KeyboardInterrupt, EOFError, SystemExit):
        break
    
    
#initializing the bot and run it with the chatterbot dependency. Needs more research. Add Functionality for twitch and stand alone Application(server side)

#training data needed (specifically for Vtuber culture and Twitch Culture)

#dependencies Python3.7./Python3.8/Conda

#Work on UI 