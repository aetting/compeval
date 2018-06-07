import nltk
import itertools
import argparse
import pickle
import os
import json
from dataset_helpers import *
from get_syntax import *
from get_event_population import *
# from dataset_lexvars import *
from dataset_dicts import *
from dataset_events import *

#takes NEEDED and PROHIBITED elements of meaning representation, both in event and list forms
#produces structures that can hold it and runs function populating each of those structures in various ways
def get_structures(task,needEvent,needList,avoidEvent,avoidList,lexvar_package,max_per_op=None,nx=False,dv=False):
    frs = [k for k in verbs]
    role_rc_structures = {}
    #make sure frames are populated if names are specified in input events
    if needEvent:
        if needEvent.name and not needEvent.frame: needEvent.frame = frames[needEvent.name]
    if avoidEvent:
        for avEv in avoidEvent:
            if avEv.name and not avEv.frame: avEv.frame = frames[avEv.name]

    if avoidList:
        for n in avoidList['noun']: nouns.remove(n)
        for t in avoidList['transitive']: verbs['transitive'].remove(t)
        for i in avoidList['intransitive']: verbs['intransitive'].remove(i)

    if task in role_rc_structures_dict: role_rc_structures = role_rc_structures_dict[task]
    else: role_rc_structures = role_rc_structures_dict['other']

    needsrc = {}
    if needEvent:
        for part in needEvent.participants:
            if 'rc' in needEvent.participants[part].attributes:
                needsrc[part] = []
                if needEvent.participants[part].attributes['rc']['event'].frame:
                    needsrc[part].append(needEvent.participants[part].attributes['rc']['event'].frame)
                else:
                    needsrc[part] += ['transitive','intransitive']


    #iterate through list, make branches to be filled, and fill them
    for f in role_rc_structures:
        mainok = True
        #if an event constraint (need or avoid) is defined, check whether frame definition eliminates this frame
        if needEvent and needEvent.frame:
            if f != needEvent.frame:
                mainok = False

        #iterate over options within this frame
        for op in role_rc_structures[f]:
            skip_op = False
            print('\n')
            print(op)
            for role in op:
                if role in needsrc and op[role] not in needsrc[role]:
                    skip_op = True
            if needEvent:
                for role in needEvent.participants:
                    if role not in op:
                        skip_op = True
            if skip_op:
                continue
#             for mainvoice in ['active','passive']:
            for event_skel,relroles in start_structured_event(op,f):
                for ev2,op,insertion,insertvoice in all_insertions(event_skel,op,needEvent,needList,avoidEvent,avoidList,max_per_op,mainok,lexvar_package,nx=nx,dv=dv):
                    yield ev2,op,relroles,insertion,insertvoice

def all_insertions(event_skel,op,needEvent,needList,avoidEvent,avoidList,max_per_op,mainok,lexvar_package,nx=False,dv=False):

    if len(needList) > 0: addfunc = populate_check_wadd
    else: addfunc = populate_check

    if needEvent:
        main_only = 0
        for role in needEvent.participants:
            if 'rc' in needEvent.participants[role].attributes: main_only = 1

        if mainok:
            #do as main event
            insertion = 'main'
            # print insertion
            for ev,main_insert_voice in insert_into_main(op,event_skel,needEvent):
                num_this_op = 0
                for ev2,num_this_op in addfunc(avoidEvent,needList,ev,max_per_op,num_this_op,lexvar_package,nx=nx,dv=dv):
                    yield(ev2,op,insertion,main_insert_voice)
#                     if max_per_op and (num_this_op >= max_per_op): break

        if not main_only:
            insertion = 'RC'
            # print insertion
            for ev,rc_insert_role,rc_insert_voice in insert_into_rcs(op,event_skel,needEvent):
                insertion = 'RC-%s'%rc_insert_role
                num_this_op = 0
                for ev2,num_this_op in addfunc(avoidEvent,needList,ev,max_per_op,num_this_op,lexvar_package,nx=nx,dv=dv):
                    yield(ev2,op,insertion,rc_insert_voice)
#                     if max_per_op and (num_this_op >= max_per_op): break

    else:
        insertion = 'non-event'
        num_this_op = 0
        for ev2,num_this_op in addfunc(avoidEvent,needList,event_skel,max_per_op,num_this_op,lexvar_package,nx=nx,dv=dv):
            yield(ev2,op,insertion,None)
#             if max_per_op and (num_this_op >= max_per_op): break


def insert_into_rcs(op,empty_event,part_event):
    for role in op:
        #rc_status is the designated frame of the relative clause we're considering for inserting the defined event
        rc_status = op[role]
        #if this role doesn't have an rc, don't bother
        if rc_status == 'none': continue
        event_structure = deepcopy(empty_event)
        if not part_event.frame or rc_status == part_event.frame:
            event_structure.participants[role].attributes['rc']['event'].absorb(part_event)
            if not event_structure.participants[role].attributes['rc']['event'].voice:
                for rc_insert_voice in ['active','passive']:
                    event_structure.participants[role].attributes['rc']['event'].voice = rc_insert_voice
                    event_structure = sync_rc_hostrole(event_structure)
                    yield event_structure,role,rc_insert_voice
            else:
                event_structure = sync_rc_hostrole(event_structure)
                yield event_structure,role,event_structure.participants[role].attributes['rc']['event'].voice
            #TODO iterate here through different options in terms of role defined role satisifer being host NP or not
            #so that's only relevant if at least (and only) one role is defined -- come to think of it same for this whole segment
        else:
            continue

def insert_into_main(op,empty_event,part_event):
    use = 1
    event_structure = deepcopy(empty_event)
    for role in part_event.participants:
        rc_status = op[role]
        #if partial event has role defined and role has an RC, check to make sure frames don't mismatch
        relfr = None
        if 'rc' in part_event.participants[role].attributes:
            if part_event.participants[role].attributes['rc']['event'].frame:
                relfr = part_event.participants[role].attributes['rc']['event'].frame
            elif part_event.participants[role].attributes['rc']['event'].name:
                relfr = frames[part_event.participants[role].attributes['rc']['event'].name]
            if relfr and relfr != rc_status:
                use = 0
                print('BREAKING BECAUSE RC MISMATCH')
                break
    if use:
        event_structure.absorb(part_event)
        if not event_structure.voice:
            for main_insert_voice in ['active','passive']:
                event_structure.voice = main_insert_voice
                event_structure = sync_rc_hostrole(event_structure)
                yield event_structure,main_insert_voice
        else:
            event_structure = sync_rc_hostrole(event_structure)
            yield event_structure,event_structure.voice
    else:
        yield None,None



def start_structured_event(op,f):
    #***currently randomly assigning relrole in rcs
    ev = eventStart({'frame':f})
    if f == 'transitive':
        relroles = [o for o in itertools.product(roles[op['agent']],roles[op['patient']])]
    else:
        relroles = [tuple([r]) for r in roles[op['agent']]]
#     print relroles

    for config in relroles:
        relroledict = {'agent':config[0]}
        if len(config) > 1: relroledict['patient'] = config[1]
        for role in op:
            rc_status = op[role]
            ev.participants[role] = characterStart()
            if rc_status != 'none':
                rcevent = eventStart({'frame':rc_status})
                rcevent = finish_role_branches(rcevent)
                relrole = relroledict[role]
#             relrole = None
                if rc_status == 'intransitive':
                    rtype = 'src'
                else:
                    rtype = None
                ev.participants[role].attributes['rc'] = {'role':relrole,'rtype':rtype,'event':rcevent}
        yield ev,relroledict

def finish_role_branches(event):
    frame = event.frame
    for role in roles[frame]:
        if role not in event.participants:
            event.participants[role] = characterStart()
    return event

def load_lexvars(lexvarfile):
    with open(lexvarfile, 'rU') as f:
        lexvars = json.load(f)
        nouns = lexvars['nouns']
        verbs = lexvars['verbs']
        frames = lexvars['frames']
        inflections = lexvars['inflections']
        nxlist = lexvars['nxlist']

    return nouns,verbs,frames,inflections,nxlist


def choose_rules(event,fixed_grammar_string,inflections):
    #choose rules with right frame for verb
    # write terminal rules for chosen participants
    grammar_string = ''
    bindings = event.bindings

    startline = get_starts(event.frame,event.voice,event.name,event.aspect,event.pol)
    grammar_string += startline

    vps = get_vp_rules(event.name,event.tense,event.pol,event.polx)
    grammar_string += vps

    vstring = get_verb_rules(event.name,inflections)
    grammar_string += vstring

    #TODO see about converting to use role rather than rc_index to identify rc in grammar
    rc_index = 1
    for role in roles[event.frame]:
        nps,rc_index,bindings = get_np_rules(role,event.participants[role],rc_index,bindings,inflections)
        grammar_string += nps

    if event.frame == 'which':
        wstring = get_wp_rules(event.wp['reltype'],event.wp['relevent'],inflections)
        grammar_string += wstring

    grammar_string += fixed_grammar_string

    tailored_grammar = nltk.grammar.FeatureGrammar.fromstring(grammar_string)
    # print(tailored_grammar)

    T = unfold_tree_feature(tailored_grammar,tailored_grammar.start(),bindings)
    return T


def write_set(task,lab,task2inputs,mpo,setdir,lexvar_package,setID=None,outname=None,nx=False,dv=False):
    # if not outname: outname = tasks[0]

    if not setID:
        fname = '%s_%s.txt'%(task,lab)
        setID=task+lab
    else: fname = setID + '.txt'
    with open(os.path.join(setdir,fname),'w') as out:
        op = None
        relroles = None
        insertion = None
        insertvoice = None
        numsent = 0
        id2ev = {}
        needEventList,needList,avoidEvent,avoidList = [task2inputs[task][lab][v] for v in ['needEv','needList','avoidEv','avoidList']]
        if needEventList and len(needEventList) > 1:
            print('DEAL WITH THIS SITUATION')
        else: needEvent = needEventList
        for ev,opNew,relrolesNew,insertionNew,insertvoiceNew, in get_structures(task,needEvent,needList,avoidEvent,avoidList,lexvar_package,max_per_op=mpo,nx=nx,dv=dv):
            if opNew != op or relrolesNew != relroles or insertionNew != insertion or insertvoiceNew != insertvoice:
                op = opNew
                relroles = relrolesNew
                insertion = insertionNew
                insertvoice = insertvoiceNew
                out.write('\n' +str(op) + '\n' + str(relroles) + '\n' + '{%s}'%insertion + '\n' + '{%s}'%insertvoice)
                out.write('\n\n')
            if ev:
                T = choose_rules(ev,gram,inflections)
                sent = ' '.join(T.leaves())
                print(sent)
                sentID = setID + str(numsent)
                out.write(sentID + '\t' + ' '.join(T.leaves()) + '\n')
                sentdict = ev.todict(inflections)
                # print(sentdict['surface'])
                # for part in ev.participants:
                #     if 'rc' in ev.participants[part].attributes:
                #         print(sentdict['participants'][part]['attributes']['rc']['event']['surface'])
                id2ev[sentID] = (sent,sentdict)
#                     print id2ev[sentID]
                numsent += 1

        out.write('\nNumber of sentences: %s\n'%str(numsent))
        with open(os.path.join(setdir,'%s-dicts.json'%(setID)),'w') as dictfile:
            json.dump(id2ev,dictfile)

if __name__ == "__main__":
    gram = import_grammar('grammar_feature')
    nouns,verbs,frames,inflections,nxlist = load_lexvars('dataset_lexvars.json')

    print("""------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------""")

    #***TODO replace command line args with config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--lab')
    parser.add_argument('--mpo',type=int,default=None)
    parser.add_argument('--setdir')
    args = parser.parse_args()

    shuffle(nxlist)
    bound = int(round(len(nxlist)*.6))
    train_nx = nxlist[:bound]
    test_nx = nxlist[bound:]

    lexvar_package = (nouns,verbs,inflections,train_nx)

    # train_lexvar_package = (nouns,verbs,inflections,train_nx)
    # test_lexvar_package = (nouns,verbs,inflections,test_nx)

    write_set(args.task,args.lab,task2inputs,args.mpo,args.setdir,lexvar_package,nx=False)
    #
    # write_set(args.task,args.lab,task2inputs,args.mpo,args.setdir,train_lexvar_package,setID='neg_train',nx=True)
    # write_set(args.task,args.lab,task2inputs,args.mpo,args.setdir,test_lexvar_package,setID='neg_test',nx=True)
