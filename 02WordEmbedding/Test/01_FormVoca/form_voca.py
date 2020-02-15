import os
os.system("pip install pandas")
import pandas as pd


df = pd.read_csv('merged.csv')
# print(df)
# print(df.iloc[:,0])
# print(df.iloc[:,0].tolist())
sentences = df.iloc[:,0].tolist()

words_count_dict = {}
for sentence in sentences:
    for word in sentence.split():
        if word in words_count_dict:
            words_count_dict[word] += 1
        else:
            words_count_dict[word] = 1

# for key in sorted(words_count_dict,key=words_count_dict.__getitem__):
#     print(key,words_count_dict[key])

# print(len(words_count_dict))

w_c_list = list(words_count_dict.items())
w_c_list.sort(key=lambda x:x[1],reverse=True)

list_file = open('vocabulary.txt','w')
w_c_list2 = list()
for word in w_c_list:
    if word[1] > 10:
        w_c_list2.append(word[0]+'\n')
# print(len(w_c_list2))
list_file.writelines(w_c_list2)