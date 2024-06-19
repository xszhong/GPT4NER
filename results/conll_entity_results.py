import numpy as np

cur_path = "./results_ontonotes_nopos_newpara.txt"
output_path = "./results_ontonotes_nopos_newpara_entity.txt"
text=""
with open (cur_path,'r',encoding='utf-8') as f:
    for line in f.readlines():
        if(line.startswith('\n')):
            text += '\n'
            continue
        ls = line.split('\t')
        if ls[1].startswith('B-'):
            text += ls[0] + '\t' + "B-ENTITY" + '\t'
        elif ls[1].startswith('I-'):
            text += ls[0] + '\t' + "I-ENTITY" + '\t'
        else:
            text += ls[0] + '\t' + "O" + '\t'

        if ls[2].startswith('B-'):
            text += "B-ENTITY" + '\n'
        elif ls[2].startswith('I-'):
            text += "I-ENTITY" + '\n'
        else:
            text += "O" + '\n'
with open (output_path,'a',encoding='utf-8') as f:
    f.write(text)
