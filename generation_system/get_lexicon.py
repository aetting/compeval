import sys
import os


#get lexical vars for generation system from XTAG
morphdir = '.'
vocabfile = 'lexicon.txt'
def get_inflections(morphdir):
    inflections = {}
    nouns = []
#     inpt = sys.argv[1]
#     out = sys.argv[2]

    f = open(os.path.join(morphdir,'morph_english.flat'), 'rU')
#     line = f.readline()
#     while (line[0] == ';'):
#         line = f.readline()

    all_inflections = {}
    noun_inflections = {}
    for line in f:
        if line[0] == ';': continue
        entry = line.split('\t')
#         pts_of_spch = []
        inflct = entry[0].rstrip()
        main_lemma = entry[2].rstrip()
        pts_of_spch = entry[3:]

#         for i in range(len(entry)):
#             if i == 0:
#                 inflct = entry[i].rstrip()
#             if i == 2:
#                 main_lemma = entry[i].rstrip()
#             if i > 2:
#                 pts_of_spch.append(entry[i])

        prog_lemmas = []
        for i in range(len(pts_of_spch)):
            if i < len(pts_of_spch)-1:
                div = pts_of_spch[i].split('#')
                lemma = div[1]
                pt_of_spch = div[0].split(' ')
            else:
                lemma = main_lemma
                pt_of_spch = pts_of_spch[i].rstrip().split(' ')

            prog_lemmas.append(lemma)
            if pt_of_spch[0] == 'V':
                # delete -ing to get lemma (works for many)
                if lemma.endswith('ing') and lemma != 'sing':
                    for lem in prog_lemmas:
                        if not lem.endswith('ing'):
                            lemma = lem
                if lemma == 'sung':
                    lemma = 'sing'

                if lemma not in all_inflections:
                    all_inflections[lemma] = {}

                if 'tensed' not in all_inflections[lemma]:
                    all_inflections[lemma]['tensed'] = {}

                if pt_of_spch[1] == 'INF':
                    all_inflections[lemma]['tensed']['prespl'] = inflct

                if pt_of_spch[1] == '3sg':
                    all_inflections[lemma]['tensed']['pressg'] = inflct

                if pt_of_spch[1] == 'PAST':
                    all_inflections[lemma]['tensed']['past'] = inflct

                if pt_of_spch[1] == 'PPART':
                    all_inflections[lemma]['pastpart'] = inflct

                if pt_of_spch[1] == 'PROG':
                    all_inflections[lemma]['prespart'] = inflct

            if pt_of_spch[0] == 'N':
                if lemma not in noun_inflections:
                    noun_inflections[lemma] = {}

                if len(pt_of_spch) < 3:
                    if pt_of_spch[1] == '3sg':
                        noun_inflections[lemma]['sg'] = inflct
                    if pt_of_spch[1] == '3pl':
                        noun_inflections[lemma]['pl'] = inflct

#         line = f.readline()
    f.close()

    return all_inflections,noun_inflections

def read_vocab(vocabfile):
    # read input file
    temp_nouns = {} # for temporarily holding nouns
    trans = {} # temporary dictionaries to remove verbs with not full inflections
    intrans = {}
    nouns_order = []

    with open(vocabfile, 'rU') as f:
        for line in f:
            cat = line.rstrip().split(' | ')
            if cat[0] == 'intransitive':
                words = cat[1].split(',')
                for word in words:
                    intrans[word.strip()] = ''
            if cat[0] == 'transitive':
                words = cat[1].split(',')
                for word in words:
                    trans[word.strip()] = ''
            if cat[0] == 'noun':
                words = cat[1].split(',')
                for word in words:
                    nouns_order.append(word.strip())
                    temp_nouns[word.strip()] = ''

    return temp_nouns,trans,intrans,nouns_order

def other(morphvars,vocabvars,out='test.py'):
    # if pastpart is the same as past tense version, just copy it over
    all_inflections,noun_inflections = morphvars
    temp_nouns,trans,intrans,nouns_order = vocabvars
    remove = []
    verbs = {'transitive':[],'intransitive':[]}
    frames = {}

    v_inflct = {} # keep nouns and verb separate at first, to see if all inflections are present
    n_inflct = {}

    past_parts = {}

    for lemma in all_inflections:
        if len(all_inflections[lemma]['tensed']) == 0:
            past_parts[lemma] = lemma

    for past_part in past_parts:
        all_inflections.pop(past_part)

    for lemma in all_inflections:
        if 'pastpart' not in  all_inflections[lemma]:
            if 'past' in all_inflections[lemma]['tensed'] and all_inflections[lemma]['tensed']['past'] in past_parts:
                all_inflections[lemma]['pastpart'] = past_parts[all_inflections[lemma]['tensed']['past']]


    # check if words have all inflections
    for v in trans:
        if v in all_inflections:
            v_inflct[v] = all_inflections[v]

    for v in intrans:
        if v in all_inflections:
            v_inflct[v] = all_inflections[v]

    for n in temp_nouns:
        if n in noun_inflections:
            n_inflct[n] = noun_inflections[n]

    errors = '# The following lemmas were not added to data structures: '

    for l in v_inflct:
        if ('tensed' not in v_inflct[l]) or ('pastpart' not in v_inflct[l]) or ('prespart' not in v_inflct[l]) or (('tensed' in v_inflct[l]) and (('pressg' not in v_inflct[l]['tensed']) or ('prespl' not in v_inflct[l]['tensed']) or ('past' not in v_inflct[l]['tensed']))):
            remove.append(l)
            if l in trans:
                trans.pop(l, None)
            if l in intrans:
                intrans.pop(l, None)
            errors += '\n# ' + l

    for l in n_inflct:
        if ('sg' not in n_inflct[l]) or ('pl' not in n_inflct[l]):
            remove.append(l)
            temp_nouns.pop(l, None)
            errors += '\n# ' + l

    # remove from inflct
    for l in remove:
        if l in v_inflct:
            v_inflct.pop(l, None)
        if l in n_inflct:
            n_inflct.pop(l, None)

    # fix random errors
    for l in n_inflct:
        if n_inflct[l]['sg'] != l:
            n_inflct[l]['sg'] = l
        if l == 'machine':
            n_inflct[l]['pl'] = 'machines'
        if l == 'peacock':
            n_inflct[l]['pl'] = 'peacocks'
        if l == 'giraffe':
            n_inflct[l]['pl'] = 'giraffes'
        if l == 'ostrich':
            n_inflct[l]['pl'] = 'ostriches'

    # build data structures
    for v in trans:
        verbs['transitive'].append(v)

    for v in intrans:
        verbs['intransitive'].append(v)

    # for n in temp_nouns:
    #    nouns.append(n)

    nouns = [n for n in nouns_order if n in temp_nouns]

    for i in verbs:
        for v in verbs[i]:
            frames[v] = i

    inflections = dict(v_inflct)
    inflections.update(n_inflct)

    with open(out, 'w') as output:
#     sys.stdout = output

        output.write('nouns = %s\n\n'%nouns)
        output.write('verbs = %s\n\n'%verbs)
        output.write('frames = %s\n\n'%frames)
        output.write('inflections = %s\n\n'%inflections)

    print(errors)

    warnings = '# The following entries may be incorrect: '
    for n in n_inflct:
        if not n_inflct[n]['pl'].endswith('s') or n_inflct[n]['pl'].endswith('es'):
            warnings += '\n# lemma: ' + n + ' pl: ' + n_inflct[n]['pl']

    print(warnings)

    return nouns,verbs,frames,inflections

if __name__ == "__main__":

    morphvars = get_inflections(morphdir)
    vocabvars = read_vocab(vocabfile)
    other(morphvars,vocabvars)
#     nouns,verbs,frames,inflections = main()
#     print nouns
#     print inflections
