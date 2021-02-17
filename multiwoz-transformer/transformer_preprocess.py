import csv

def getDialogs(max_len = 0, file = 'train'):
    reader = csv.reader(open('MultiWoZ_'+file+'.csv','r'))
    dialog = {}
    id_ = 1
    prev_dial = None
    prev_row = None

    for row in reader:
         if prev_dial != None and prev_dial != row[2]:
             st = prev_row[5] + '<eou>' + '<eou>'.join(reversed(prev_row[4].split('<eou>')[:-1]))+'<eou>'
             l = len(st.split('<eou>'))
             st += '<pad> <eou>'*max(0,max_len-l)
             dialog.update({id_:st})
             id_+=1
         prev_dial = row[2]
         prev_row = row

    return dialog

def writeDialogs(file = 'train'):
    print(file)
    target = open('multiwoz_trans_'+file+'.csv', "w")
    dialog = getDialogs(max_len = 0, file = file)
    l = max([len(dialog[i+1].split('<eou>')) for i in range(0,len(dialog))])
    dialog = getDialogs(max_len = l, file = file)
    fieldnames = ['response','context']+['context/'+str(i) for i in range(l-2)]
    writer = csv.DictWriter(target, fieldnames=fieldnames[:-1])

    for _,dial in dialog.items():
         d = dict([(f,u.strip()) for f,u in zip(fieldnames,dial.split('<eou>')[:-1])])
         writer.writerow(d)
    print(file+' done.')

if __name__ == '__main__':
    writeDialogs(file = 'train')
    writeDialogs(file = 'valid')
    writeDialogs(file = 'test')
