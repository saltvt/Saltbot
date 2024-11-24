import sys
import twitchio
import os
import tweepy
import discord
import json
from atproto import Client
import saltNN



def isRunning():
    print(saltNN.saltBotRunning)
    return saltNN.saltBotRunning

if isRunning()==False:
    sys.exit(1)
#focus plugging in integrations
#adding discord, bluesky, discord, twtich integrations. Will add more later depending on the next couple of years 
#need to create an application to get this information
#Will probably push this and all login info to auth later so that it's not stored in plain text
#will definitely be slow but I'll optimize

name = "Saltbot App"
twitchID="Saltvt"
bskyID= Client()



twitterApi = tweepy.Client(
    consumer_key="",
    access_token="",
    access_token="",
    consumer_secret="",
)

discord.IntegrationApplication(name)
twitchio.Channel(twitchID)


#commands the bot can use to post/interact with the world. Made it an empty array because it should be ever expanding. Optimization comes later. 
commands=[]