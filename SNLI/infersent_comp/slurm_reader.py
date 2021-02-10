## Read output from slurm and create a csv file

import pandas as pd
import glob

if __name__ == '__main__':
    # list all the slurm outputs
    results = glob.glob('*.out')
    rows = []
    for res in results:
        r_file = open(res).readlines()
        data_file = ''
        model = ''
        test_acc = ''
        val_acc = ''
        seed = ''
        for rf in r_file:
            if 'Cmd:' in rf:
                seed = rf.split('--seed')[-1]
            if "togrep : ['--nlipath'" in rf:
                sp = rf.split(',')
                data_file = sp[1]
                model = sp[3].lstrip().replace("'",'')
                continue
            if "finalgrep : accuracy valid :" in rf:
                sp = rf.split(" : ")
                val_acc = sp[-1].rstrip()
                continue
            if "finalgrep : accuracy test :" in rf:
                sp = rf.split(" : ")
                test_acc = sp[-1].rstrip()
                rows.append({'data_file': data_file, 'model':model, 'val_acc':val_acc,'test_acc':test_acc, 'seed': seed})
    df = pd.DataFrame(rows)
    df.to_csv('all_results.csv')
