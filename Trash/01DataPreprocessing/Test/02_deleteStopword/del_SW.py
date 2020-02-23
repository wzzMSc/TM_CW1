import os
os.system("pip install stop-words")
from stop_words import get_stop_words
stop_words = get_stop_words('english')

input = open('train_5500_sentences.txt','r')
output = open('train_5500_sentences_noSW.txt','w')

outlines = list()
for line in input.readlines():
    linelist = line.split()
    for word in linelist:
        if word in stop_words:
            linelist.remove(word)
    outlines.append(' '.join(linelist)+'\n')


output.writelines(outlines)
input.close()
output.close()