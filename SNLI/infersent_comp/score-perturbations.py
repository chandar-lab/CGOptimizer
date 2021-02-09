import glob
import re
import os
import cProfile
from collections import defaultdict
from sentence import Sentence
from item import Item
import argparse
import pandas as pd
from tqdm import tqdm
import itertools
from itertools import combinations
import operator as op
from functools import reduce
import multiprocessing
from multiprocessing import Pool
import concurrent.futures
import copy

test_directory=""
# Specify the number of processes here
num_proc = 32


sum_list=defaultdict(int)
length_list=defaultdict(int)

def iterator_slice(iterator, length):
    iterator = iter(iterator)
    while True:
        res = tuple(itertools.islice(iterator, length))
        if not res:
            break
        yield res

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def comb(list1, list2):
    for i in range(len(list1)):
        for j in range(i+1, len(list2)):
             yield(list1[i], list2[j])

def prune_dist(list_test_pairs):
    big_list, sm_list, test_pairs = list_test_pairs
    #print(os.getpid())
    pruned_pairs = []
    #print(len(big_list))
    #print(len(sm_list))
    for pair in comb(sm_list, big_list):
        i = test_pairs[pair[0]]
        j = test_pairs[pair[1]]
        d = Item.distance(i, j)
        if d == 1:
            pruned_pairs.append(pair)
    return pruned_pairs



def calculate_stats(step):
    print(step)
    output_files = glob.glob(step + '/*outp.csv')
    pbc = tqdm(total=len(output_files))
    for outp_file in output_files:
        rows = []
        outfile = pd.read_csv(outp_file)
        test_pairs = {}
        #print("Precomputing the sentences ... ")
        for row_indx,row in outfile.iterrows():
            premise=row['s_1'].replace('<s>','').replace('</s>','').strip()
            hypothesis=row['s_2'].replace('<s>','').replace('</s>','').strip()
            correct=row['true_target']
            label=row['predicted']
            prem=Sentence.parse(premise)
            hyp=Sentence.parse(hypothesis)

            i=Item(prem,hyp,correct,label,True)
            if not(i.is_target()): continue
            test_pairs[row_indx] = i
            #print(row_indx, end = "\r")
        #print("\n")
        #print("Done, found {} targets".format(len(test_pairs)))
        
        big_list = list(test_pairs.keys())
        small_lists = list(split(big_list, num_proc))
        #print("Big list with {} items".format(len(big_list)))
        small_lists = [(big_list, sl, test_pairs) for sl in small_lists]
        #print("Number of small lists = {}".format(len(small_lists)))
        # Precompute the pairs whose distance = 1
        ct = 0
        num_comp = nCr(len(test_pairs),2)
        #print("Evaluating {} combinations".format(num_comp))


        d_pairs = []
        #with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        #    for pair in executor.map(prune_dist, combinations(test_pairs.values(), 2)):
        #        if pair:
        #            d_pairs.append(pair)

 
        try:
            pool = Pool(num_proc)
            d_pairs = pool.imap_unordered(prune_dist, small_lists) 
        finally:
            pool.close()
            pool.join()
        #print(ct, end="\r")
        #print("\n")
        d_pairs = [y for x in d_pairs for y in x]
        #print("pairs computed, found {}".format(len(d_pairs)))

        # preprocessed each sentence, now calculate the statistics       
        for pair in d_pairs:
            i = test_pairs[pair[0]]
            j = test_pairs[pair[1]]
            diff_words = str(Item.diff(i,j))
            before_sub_correct = str(int((i.correct == i.label)))
            after_sub_correct = str(int(j.correct == j.label))
            bool_ct = before_sub_correct + after_sub_correct
            rows.append([step, outp_file, pair[0], pair[1], diff_words, bool_ct, i.correct, i.label, j.correct, j.label])
        df = pd.DataFrame(rows, columns=["Step","Filename","Pair A Index","Pair B Index","Diff","Boolean","Pair A Correct Label","Pair A Predicted Label","Pair B Correct Label","Pair B Predicted Label"])
        step = outp_file.split('/')[-1].split('.csv')[0]
        exp_folder = outp_file.split('/')[0]
        scratch_exp_folder = os.path.join('/scratch/koustuvs/compnli/', exp_folder)
        if not os.path.exists(scratch_exp_folder):
            os.mkdir(scratch_exp_folder)
        print("Saving to folder : {}".format(scratch_exp_folder))
        df.to_csv(scratch_exp_folder + '/' + step + '_statistics.csv')
        pbc.update(1)
    pbc.close()
    
def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--exp_name", type=str, default='exp_seed_11', help='Experiment name') 
    parser.add_argument("-s","--step_name", type=str, default='stepH0', help='step name, comma separared')
    args = parser.parse_args()
    dirs = glob.glob(args.exp_name + '/step**')
    steps = args.step_name.split(',')
    dirs = [d for d in dirs if d.split('/')[-1] in steps]
    print('Steps found : {}'.format(len(dirs)))
    pb = tqdm(total=len(dirs))
    for step in dirs:
        calculate_stats(step)
        pb.update(1)
        break
    pb.close()
