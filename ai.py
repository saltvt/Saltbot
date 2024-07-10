import chatterbot
import tkinter
import twitchio pip install
#import chatterbot 


class Bot(commands.bot);

def __init__(self);
    super().__init(token='YOUR_0AUTH_TOKEN', prefix='!', initial_channels=['Twitch.tv/Saltvt'])
    
    async def even_ready(self):
        print (f'Logged in as | {self.nick}')
        
    @commands.command()
    async def hello(self, ctx: commands.Context):
        await ctx.send('Hello {ctx.author.name}')
        
        
bot=Bot()
bot.run()
