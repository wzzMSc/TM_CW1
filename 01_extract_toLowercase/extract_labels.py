input = open('train_5500.label','r')
output = open('train_5500_labels.txt','w')
outlines = list()
for line in input.readlines():
    outlines.append( line.split()[0]+'\n' )
output.writelines(outlines)
input.close()
output.close()