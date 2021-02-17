## Read output from slurm and create a csv file

import pandas as pd
import glob

if __name__ == '__main__':
    # list all the slurm outputs
    results = glob.glob('step*.txt')
    rows = []
    for res in results:
        r_file = open(res).readlines()
        data_file = res.split('_')[0]
        model = res.split('_')[1]
        if model.isdigit():
            data_file = res.split('_')[0] + '_' + res.split('_')[1]
            model = res.split('_')[2]
        test_acc = r_file[1].split(' - ')[-1].rstrip()
        val_acc = r_file[0].split(' - ')[-1].rstrip()
        rows.append({'data_file': data_file, 'model':model, 'val_acc':val_acc,'test_acc':test_acc})
    df = pd.DataFrame(rows)
    df.to_csv('all_results.csv')
