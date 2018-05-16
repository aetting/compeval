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
    return corrlist

def bootstrap(orig_sample,num_samples):
    n = len(orig_sample)
    # orig_acc = sum(orig_sample)/float(n)
    # diffs = []
    # for i in range(num_samples):
    #     resample with replacement to get sample size n
    #     acc = sum(sample)/float(n)
    #     diff = acc - orig_acc
    #     diffs.append(diff)
    # sort diffs
    # select diffs at 2.5 and 97.5 percentiles (or set up to actually calculate that) as endpoints
    # return low_end,high_end


# get size n of sample
# resample with replacement to get new sample of size n
# compute accuracy
# get diff between real sample accuracy and new sample accuracy
# append to deviation list
# repeat number of sample times (eg 10000)
#sort differences
#take endpoints at appropriate percentiles (2.5 and 97.5)

if __name__ == "__main__":
    read_results_file('../dataset/loc_results.txt')
