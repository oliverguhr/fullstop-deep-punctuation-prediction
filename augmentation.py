import os
import re
import codecs
import pandas as pd

class Token(object):
    
    def __init__(self, literal, output, first_task, second_task):
        #Original im Text
        self.literal = literal
        #lowercased
        self.output = output
        #daten für satzsegmentierung
        self.first_task = first_task
        #daten für subtask 2
        self.second_task = second_task

class Augmentation(object):

    def __init__(self, rootdir='C:/Users/Anne/Desktop/korpora/bundestagsdebatten/sessions/csv', leipzigdir='C:/Users/Anne/Desktop/test'):
        self.rootdir = rootdir
        self.leipzigdir = leipzigdir
        self.lines = []
        self.tokens = []

    def read_df(self):
        outfile = codecs.open('bundestag_aug.txt', 'wb', 'utf8')
        for subdir, dirs, files in os.walk(self.rootdir):
            for file in files:
                df = pd.read_csv(os.path.join(self.rootdir, file), index_col=0)
                data = df[~(pd.isnull(df['speaker_key']))]
                for item in data['text'].astype('str').values:
                    #utterances = re.split(r'(\n)+', item)
                    #vielleicht nicht durch \n splitten, um den größeren dokumentkontext zu behalten
                    #utterances = list(filter(lambda x: x != '\n', utterances))
                    #self.lines.extend(utterances)
                    text = re.sub(r'\n', ' ', item)
                    self.lines.append(text)
        for line in self.lines:
            for i, token in enumerate(line.split()):
                literal = token
                output = ''.join([i for i in literal.lstrip('"„').lower() if i.isalnum()])
                first_task = 0
                second_task = ''.join([i for i in literal[-1] if not i.isalnum()])
                if (not second_task or len(second_task) == 0) and i < len(line.split())-1:
                    second_task = ''.join([i for i in line.split()[i+1][0] if not i.isalnum()])
                print("{}\t{}\t{}".format(output, first_task, second_task), file=outfile)
                #self.tokens.append(Token(literal, output, first_task, second_task))

    def read_leipzig(self):
        leipzig1 = codecs.open(os.path.join(self.leipzigdir, 'deu_news_2015_1M-sentences.txt'), 'rb', 'utf8')
        leipzig2 = codecs.open(os.path.join(self.leipzigdir, 'deu_mixed-typical_2011_1M-sentences.txt'), 'rb', 'utf8')
        lines = leipzig1.readlines()
        lines.extend(leipzig2.readlines())
        leipzig1.close()
        leipzig2.close()
        for line in lines:
            items = re.split(r'\t', line)
            try:
                #wir entfernen ein paar Zeichen, bei denen wir uns nicht sicher sind
                text = re.sub(r'[-–&]', '', items[1])
                for i, token in enumerate(text.split()):
                    literal = token
                    output = ''.join([i for i in literal.lstrip('"„').lower() if i.isalnum()])
                    first_task = 0 if i < len(text.split())-1 else 1
                    second_task = ''.join([i for i in literal[-1] if not i.isalnum()])
                    #catch "" of next word
                    if (not second_task or len(second_task) == 0) and first_task == 0:
                        second_task = ''.join([i for i in text.split()[i+1][0] if not i.isalnum()])
                    self.tokens.append(Token(literal, output, first_task, second_task))
            except Exception:
                print(items)
                