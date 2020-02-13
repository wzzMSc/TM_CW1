input = open('train_5500.label','r')
output = open('train_5500_sentences.txt','w')
outlines = list()
for line in input.readlines():
    outlines.append( ' '.join(line.lower().split()[1:-1])+'\n' )
output.writelines(outlines)
input.close()
output.close()