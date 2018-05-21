import numpy as np

nouns = [('ant','bat'),('bat','cat'),('dog','ant')]
trans = [('chase',),('eat',),('see',)]
intrans = ['sleep','dance']
x = [1,2]

permns = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
# (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)]
permts = [(0,), (1,), (2,), (3,), (4,)]
permis = [()]
tenses = ['past','pres']

# Ns = [available_nouns[e] for e in permn]
# Ts = [available_verbs['transitive'][e] for e in permt]
# Is = [available_verbs['intransitive'][e] for e in permi]

permns = ['1','2','3']
permts = ['a','b']
permis = ['x','y','z']
voices = ['c','d']
tenses = ['l','m']
aspects = ['q','r']
pols = ['j','k']

# def check(combs,used):
#     if ncomb not in used:
#         return safe
#     else:
#         if allverbs not in used[ncomb]:
#             return safe
#         else:
#             if other not in used[ncomb][allverbs]:
#                 return safe
#             else:
#                 return notsafe

def decode(num,listlengths):

    inds = []
    for i,l in enumerate(listlengths):
        if i == 0:
            ind = num/np.prod(listlengths[1:])
            # permn_ind = num/(lt*li*lv*ltn*la*lp)
        elif i == len(listlengths) - 1:
            ind = num%listlengths[-1]
        else:
            ind = (num%(np.prod(listlengths[i:])))/(np.prod(listlengths[i+1:]))
        inds.append(ind)
    return inds

def get_rand_prod(lists,maxnum):
    used = []
    listlengths = [len(l) for l in lists]
    numcombs = np.prod(listlengths)
    new = 0
    for i in range(maxnum):
        while new == 0:
            num = np.random.randint(numcombs)
            if num not in used:
                new = 1
        inds = decode(num,listlengths)
        listitems = [lists[i2][ind2] for i2,ind2 in enumerate(inds)]
        # s2 = ' '.join(listitems) + ' --> %s'%num
        used.append(num)
        new = 0
        yield listitems


if __name__ == "__main__":
    lists = [permns,permts,permis,voices,tenses,aspects,pols]
    # listlengths = [len(l) for l in lists]
    for permn,permt,permi,voice,aspect,tense,pol in get_rand_prod(lists,5):



    # ind = 0
    # for i in permns:
    #     for ii in permts:
    #         for iii in permis:
    #             for v in voices:
    #                 for t in tenses:
    #                     for a in aspects:
    #                         for p in pols:
    #                             s1 = '%s %s %s %s %s %s %s --> %s'%(i,ii,iii,v,t,a,p,ind)
    #                             inds = decode(ind,listlengths)
    #                             s2 = ' '.join([str(lists[i2][ind2]) for i2,ind2 in enumerate(inds)]) + ' --> %s'%ind
    #                             if s1 != s2: print('%s %s'%(s1,s2))
    #                             ind += 1
