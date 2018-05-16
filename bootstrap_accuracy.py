import numpy as np
import re

def read_results_file(filename):
    corrlist = []
    with open(filename,'rU') as f:
        on = False
        for line in f:
            if re.match('ID',line):
                on = True
                continue
            if not on: continue
            lsplit = line.strip().split('\t')
            if len(lsplit) < 2: continue
            corr = int(lsplit[3])
            corrlist.append(corr)
    return np.array(corrlist)

def bootstrap(orig_sample,num_samples):
    n = len(orig_sample)
    orig_acc = sum(orig_sample)/float(n)
    diffs = []
    for i in range(num_samples):
        samp = orig_sample[np.random.randint(0,n,n)]
        acc = sum(samp)/float(n)
        diff = acc - orig_acc
        # print(diff)
        diffs.append(diff)
    diffsort = sorted(diffs)
    # print(diffsort)
    lowind = int(round(.025*len(diffsort)))
    highind = int(round(.975*len(diffsort)))
    lowdiff = diffsort[lowind]
    highdiff = diffsort[highind]

    low = orig_acc + lowdiff
    high = orig_acc + highdiff

    return orig_acc,low,high

def file2bootstrap(filename,num_samples):
    corrlist = read_results_file(filename)
    acc,low,high = bootstrap(corrlist,num_samples)

    return acc,low,high

# get size n of sample
# resample with replacement to get new sample of size n
# compute accuracy
# get diff between real sample accuracy and new sample accuracy
# append to deviation list
# repeat number of sample times (eg 10000)
#sort differences
#take endpoints at appropriate percentiles (2.5 and 97.5)

if __name__ == "__main__":
    filename = '../dataset/loc_results.txt'
    acc,low,high = file2bootstrap(filename,10000)
    print('%s < %s < %s'%(low,acc,high))
