import seaborn as sns
sns.set()
import pandas as pd
import csv
import os
import argparse
import matplotlib.pyplot as plt
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'wikitext-2')
parser.add_argument('--model', default = 'LR')
#parser.add_argument('--graphparam', default = 'METEOR')
args = parser.parse_args()

# requires csv with headers alpha values, epochs, meteor
def createData(dataset, Opts, model): # alpha->exp mapping {1:'1E-1',2:'1E-3'}
    #mapdict = {20:'Baseline', 22: 'BERT', 30: 'fastText', 31: 'GloVe'}#{2:'Baseline', 1:'1E-3',4:'1E0'}#{2:'Baseline', 1: '1E-3', 0: '1E-2',3: '1E-1', 4: '1E0', 5:'1E1', 6:'1E2'}
    seeds = ['100','101','102','103','104']
    #lr = ['0.0001','0.001','0.01']
    lr = {'Adam':'0.001','SGD':'0.01','SGDM':'0.01'}
    topC = ['3']
    kappa = ['0.99']
    fieldnames=['Model','Dataset','Optimizer','Seed','Epoch','Train Loss','Val. Loss','Val. PPL']
    target = open("results_valid.csv", "w")
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writerow(dict(zip(fieldnames, fieldnames)))
    for v in Opts:
        print('Now gathering info from ',v)
        for s in seeds:
            print(s)
            #if '_C' in v:
            #    folder = os.path.join('..','Results',args.dataset, model +'_'+ v, 'Model','seed_'+s+'_LR_'+l+'_topC_'+c+'_kappa_'+k)
            #    opt = v+c
            #else:
            folder = os.path.join('..','Results',args.dataset, model+'_' + v, 'Model','seed_'+s+'_LR_'+lr[v])
            opt = v
            f = open(os.path.join(folder,'logs.txt'))
            ep = 1
            for line in f:
                line = line.split('|')
                if line !=['\n']:
                    writer.writerow(dict([
                    ('Model',model),
                    ('Dataset',args.dataset),
                    ('Optimizer',opt),#+' '+l),
                    ('Seed',s),
                    ('Epoch',ep),
                    ('Train Loss',line[2].split()[-1]),
                    ('Val. Loss',line[3].split()[-1]),
                    ('Val. PPL',line[4].split()[-1])])
                    )

                    ep+=1
    target.close()
    return fieldnames

def createGraph(yrange=[0.0,0.5], model = 'model', dataset = 'PTB',filename = 'results_valid.csv', graphparam = 'METEOR', save_file = 'test',opt = None):
    #plt.ylim(yrange[0],yrange[1])
    #if 'oss' in graphparam:
     #   plt.ylim(yrange[0],yrange[1])
    sns.lineplot(x = 'Epoch', y =graphparam, hue = 'Optimizer',data = pd.read_csv(filename))
    if opt != None:
        plt.title('Perf. of '+model+' on ' +dataset + ' dataset with Optimizer as ' + opt[0])
    else:
        plt.title('Perf. of '+model+' on ' +dataset + ' dataset')
    plt.savefig(save_file)
    plt.close()

if __name__ == '__main__':
    Opts = [['Adam'],['SGD'],['SGDM']]#,'SGDM','SGD']]#,('Adam_C','Adam')]#,('SGDM_C','SGDM')]
    models = ['LSTM']
    for model in models:
        print(model)
        for opt_ in Opts:
            header = createData(args.dataset, model = model, Opts = opt_)
            for param in header[5:]:
                createGraph(graphparam = param, model = model, save_file = model+'_'+param + '_'.join(opt_)+'.png', opt = opt_)
