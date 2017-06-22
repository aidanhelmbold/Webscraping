import glob
import csv
import string
from collections import defaultdict

def clean_line(line):    
    def _is_not_punct(char):
        return char not in '"#$%&\'();*+-./<=>?@[\\]^_`{|}~' + '\n\t\r'
    """
    Preprocesses text for training through string methods.
    """
    line = line.strip()
    line = line.lower()
    line = line.replace('mr.', 'mr').replace('mrs.', 'mrs').replace('ms.', 'ms').replace('etc.', 'etc')
    line = ''.join(filter(_is_not_punct, line))
    line = line.replace(',', ' ,')
    line = line.replace(':', ' :')
    line = line.replace(';', ' ;')
    line = line.replace('!', ' !')
    line = line.split()
    line.insert(0, '^')
    line.append('.')
    return line

def open_lyrics(artist):
    csv_file = 'data/' + artist +'.csv'
    try:
        return csv.reader(open(csv_file, mode='r'))
    except:
        print ('{:s} lyrics file does not exist'.format(artist) )
        return None
    

def map_lyrics(artist):
    word2idx = {'^': 0, '.': 1}
    word_count = defaultdict(int)
    current_idx = 2
    sentences = []    
    
    reader = open_lyrics(artist)
    
    for song in reader:
        for line in song[1].split('\n'):
            tokens = clean_line(line)
            sentence = []
            for t in tokens:
                word_count[t] += 1
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx, word_count