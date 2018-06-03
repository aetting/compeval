import os
import re
import argparse
from bootstrap_accuracy import *
import matplotlib.pyplot as plt

# setdir = '/fs/clip-scratch/aetting/sets/'
#setdir = '../sets/'

trainfolds = 4

# tasks = ['hasprof','profhelp','profhelppat','neghelp']
tasks={'cont1':'Content1Probe','cont2':'Content2probe','order':'Order','sr':'SemRole','neg':'Negation'}

emb_methods = ['bow','sdae','skipthoughts-uni','skipthoughts-bi','infersent']
plotnames = ['BOW','SDAE','ST-U','ST-B','INF']
#emb_methods = ['bow','sdae','skipthoughts-uni','infersent']
# emb_methods = ['bow']

def get_acc_sd(repfile):
    with open(repfile) as rep:
        for line in rep:
            accmat = re.match('MEAN ACCURACY: (.*)',line.strip())
            sdmat = re.match('STD DEV: (.*)',line.strip())
            if accmat:
                acc = accmat.group(1).strip()
            if sdmat:
                sd = sdmat.group(1).strip()
    return (acc,sd)

def get_xy_acc(repfile):
    with open(repfile) as rep:
        for line in rep:
            accmat = re.match('classification accuracy: (.*)',line.strip())
            if accmat:
                acc = accmat.group(1).strip()
                break
    return acc

def get_all(datadir,splitname,cltype):
    resdict = {}
    for task in tasks:
        resdir = os.path.join(datadir,task,splitname,'results')
        resdict[task] = {}
        taskreport = os.path.join(resdir,'allreport-%s.txt'%cltype)
        with open(taskreport,'w') as rep:
            for meth in emb_methods:
                rep.write('MODEL: ' + meth.upper() + '\n\n')
                resfile = os.path.join(resdir,meth,'full_report-%s.txt'%cltype)
                resdict[task][meth] = get_acc_sd(resfile)
                f = open(resfile)
                for line in f:
                    rep.write(line)
                f.close()
                rep.write('\n\n\n')
    write_latex_table(datadir,resdict,splitname,cltype)

def get_all_xy(datadir):
    resdict = {}
    for task in tasks:
        resdir = os.path.join(datadir,task)
        resdict[task] = {}
        taskreport = os.path.join(resdir,'allreport.txt')
        olhlist = []
        with open(taskreport,'w') as rep:
            for meth in emb_methods:
                rep.write('MODEL: ' + meth.upper() + '\n\n')
                resfile = os.path.join(resdir,meth,'full_results.txt')
                resdict[task][meth] = get_xy_acc(resfile)
                f = open(resfile)
                for line in f:
                    rep.write(line)
                f.close()
                itemfile = os.path.join(resdir,meth,'loc_results.txt')
                orig_acc,acclow,acchigh = file2bootstrap(itemfile,10000)
                olhlist.append((orig_acc,acclow,acchigh))
                rep.write('\n\nCONFIDENCE INTERVALS (LOCALIST)\n\n%s < %s < %s\n'%(acclow,orig_acc,acchigh))
                rep.write('\n\n\n')
            plot_cis(olhlist,task,resdir)
    write_latex_table_xy(datadir,resdict)

def plot_cis(olhlist,task,resdir):
    # olhlist = ((3.3,3.0,3.6),(3.2,2.8,3.5),(3.3,3.0,3.6),(3.2,2.8,3.5),(3.3,3.0,3.6))
    pts = [o for o,_,_ in olhlist]
    low = [o-l for o,l,_ in olhlist]
    high = [h-o for o,_,h in olhlist]
    x = range(1,len(pts)+1)
    plt.scatter(x,pts,c='green',s=50,marker='_',linewidths=2)
    plt.errorbar(x,pts,yerr=np.array([low,high]),ecolor='black',linewidth=1,fmt='none')
    # plt.xlim(0,len(diffs)+1)
    # plt.ylim(ymin=-.02)
    plt.xticks(tuple(x),plotnames)
    plt.xlabel("Embedding method")
    plt.ylabel("Accuracy")
    plt.title('%s accuracies'%tasks[task])
    plt.savefig(os.path.join(resdir,'%s-cis.png'%task),format='png')
    plt.clf()


def write_latex_table(outdir,resdict):
    file = os.path.join(outdir,'%s-textable-%s.tex')
    with open(file,'w') as out:
        out.write('\\begin{table*}\n')
        cs='c'*(len(tasks)+1)
        out.write('\\centering\n\\begin{tabular}{%s|}\n'%cs)
        out.write('\\cline{2-%s}\n'%(len(tasks)+1))
        out.write('& \\multicolumn{%s}{| c |}{Accuracy}  \\\\ \n'%len(tasks))
        for i,task in enumerate(tasks):
            if i == 0: colstr = '| c'
            elif i == (len(tasks)-1): colstr = 'c |'
            else: colstr = 'c'
            out.write('&  \\multicolumn{1}{%s}{%s}\n'%(colstr,task))
        out.write('\\\\ \\hline\n')
        for meth in emb_methods:
            out.write('\\multicolumn{1}{| c |}{%s}'%meth)
            for task in tasks:
                acc,sd = resdict[task][meth]
                out.write('& $%0.3f \\pm %0.3f$'%(float(acc),float(sd)))
            out.write('\\\\\n')
        out.write('\\hline\n')
        out.write('\\end{tabular}')
        out.write('\\caption{*%s* training folds} \\label{tab:%sfoldall}\n'%(trainfolds,trainfolds))
        out.write('\\end{table*}')

def write_latex_table_xy(outdir,resdict):
    file = os.path.join(outdir,'textable.tex')
    with open(file,'w') as out:
        out.write('\\begin{table*}\n')
        cs='c'*(len(tasks)+1)
        out.write('\\centering\n\\begin{tabular}{%s|}\n'%cs)
        out.write('\\cline{2-%s}\n'%(len(tasks)+1))
        out.write('& \\multicolumn{%s}{| c |}{Accuracy}  \\\\ \n'%len(tasks))
        for i,task in enumerate(tasks):
            if i == 0: colstr = '| c'
            elif i == (len(tasks)-1): colstr = 'c |'
            else: colstr = 'c'
            out.write('&  \\multicolumn{1}{%s}{%s}\n'%(colstr,task))
        out.write('\\\\ \\hline\n')
        for meth in emb_methods:
            out.write('\\multicolumn{1}{| c |}{%s}'%meth)
            for task in tasks:
                acc = resdict[task][meth]
                out.write('& $%0.1f$'%(100*float(acc)))
            out.write('\\\\\n')
        out.write('\\hline\n')
        out.write('\\end{tabular}')
        out.write('\\caption{*%s* training folds} \\label{tab:%sfoldall}\n'%(trainfolds,trainfolds))
        out.write('\\end{table*}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('traincats',type=int)
    # parser.add_argument('testcats',type=int)
    # parser.add_argument('splitt')
    parser.add_argument('datadir')
    # parser.add_argument('cltype')
    args = parser.parse_args()

    # traincats = args.traincats
    # testcats = args.testcats
    # splitname = args.splitt
    datadir = args.datadir
    # cltype = args.cltype

    ddxy = os.path.join(datadir,'xy')

    # splitname = '%s-%s-%s'%(traincats,testcats,splitt)
    # get_all(datadir,splitname,cltype)
    get_all_xy(ddxy)

# \begin{table*}
# \centering
# \begin{tabular}{cccccc|}
# \cline{2-5}
#    & \multicolumn{4}{| c |}{Accuracy}  \\
#   &  \multicolumn{1}{| c}{has-professor}
#   &  \multicolumn{1}{c}{prof-as-agent}
#   & \multicolumn{1}{c}{prof-as-patient}
#   & \multicolumn{1}{c |}{negation} \\ \hline
# \multicolumn{1}{| c |}{w2v2400} & & & & \\
#
# \multicolumn{1}{| c |}{CombinedST} & & & &\\
# \multicolumn{1}{| c |}{SDAE2400} & & &  &\\
# \multicolumn{1}{| c |}{Infersent2400} & & & &\\\hline
# \end{tabular}
# \caption{FOUR training folds}
#   \label{tab:results}
# \end{table*}

    #write file
