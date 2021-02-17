# Convert emily's data into this repository format
# this repo format is : s1.train. s2.train
#                       s1.dev, s2.dev
#                       s1.test, s2.test
#                       labels.train, labels.dev, labels.test

import argparse
import glob
import os
import random
import re

def split_sents(recs, path, dt):
    s1 = []
    s2 = []
    labels = []
    for rec in recs:
        a,b,c = rec.split('\t')
        s1.append(re.sub(' +', ' ', b.replace('(', '').replace(')','')))
        s2.append(re.sub(' +', ' ', c.replace('(', '').replace(')','')))
        labels.append(a)
    with open(os.path.join(path, 's1.{}'.format(dt)),'w') as fp:
        fp.write('\n'.join(s1))
    with open(os.path.join(path, 's2.{}'.format(dt)),'w') as fp:
        fp.write('\n'.join(s2))
    with open(os.path.join(path, 'labels.{}'.format(dt)),'w') as fp:
        fp.write('\n'.join(labels))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, default='dataset/compositionality/quantifiers/koustuv/cmb/')
    parser.add_argument('--split', type=float, default=0.8)
    args = parser.parse_args()

    # list folders
    folders = glob.glob(args.loc + '*/')
    print("Found {} folders".format(len(folders)))
    for folder in folders:
        print("Directory : {}".format(folder))
        train_fl = os.path.join(folder, 'train.txt')
        print("reading file: {}".format(train_fl))
        train = [line.rstrip('\n') for line in open(train_fl)]
        # split dev
        train_indices = random.sample(list(range(0, len(train))), int(len(train) * args.split))
        dev = [rec for idx, rec in enumerate(train) if idx not in set(train_indices)]
        train = [train[ti] for ti in train_indices]
        print("train # : {}".format(len(train)))
        print("dev # : {}".format(len(dev)))
        test_fl = os.path.join(folder, 'test.txt')
        print("reading file : {}".format(test_fl))
        test = [line.rstrip('\n') for line in open(test_fl)]
        print("test # : {}".format(len(test)))
        split_sents(train, folder, 'train')
        split_sents(dev, folder, 'dev')
        split_sents(test, folder, 'test')
        print("Done saving.")




