"""Tabulate raw signed one-step effects without retaining the counterfactual models."""
import csv
import os
from contextlib import ExitStack
from pathlib import Path
import json

from .b_panels import selection_score


def core_tables(dataset,paths,folder):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    base = ['dataset','seed','epoch','scope','family','q','population','receiver']
    schemas = {
        'B_ANCHOR_MANIFEST':['dataset','seed','epoch','global_step','B','row_hash','group_hash',
                             'pre_steps_hash','post_steps_hash','actual_joint','virtual_joint','source_file'],
        'B0_EFFECTS':base+['mode','readout','reference','status','radius','delta_auc','delta_fixed_pairs',
                          'delta_ll','source_U_auc','self_U','other_mean_U','worst_U','macro_U'],
        'B1_EMA_INTERACTIONS':base+['D_online','D_online_small','D_F_small','D_F_actual',
                                    'I_hist','D_F_full','D_F_actual_minus_D_F_small'],
        'B0_IDENTITY_CONTROLS':['dataset','seed','epoch','family','q','n_negative','n_positive',
                                'loss_mass','membership_hash','P_A_score','selected_q','NO_ACTION'],
    }
    files, writers = {},{}
    with ExitStack() as stack:
        for k,fields in schemas.items():
            path = folder/f'{k}_{dataset}.csv'
            tmp = path.with_suffix('.tmp')
            stream = stack.enter_context(tmp.open('w',newline=''))
            files[k]=(tmp,path)
            writers[k]=csv.writer(stream)
            writers[k].writerow(fields)
        for path in paths:
            r=json.loads(Path(path).read_bytes())
            i=r['identity']
            seed,epoch=i['seed'],i['epoch']
            writers['B_ANCHOR_MANIFEST'].writerow([i[k] for k in schemas['B_ANCHOR_MANIFEST'][:9]]+
                                  [i['actual_joint']['model'],i['virtual_joint']['model'],str(path)])
            for a in r['effects']:
                family,q,mode=a['family'],a['q'],a['mode']
                if mode=='LOO' and a['scope']=='EMBED':
                    part=r['partitions']
                    writers['B0_IDENTITY_CONTROLS'].writerow([dataset,seed,epoch,family,q,
                        *part['capacities'][q],part['losses'][family][q],part['membership_hashes'][family],
                        selection_score(a['online']),r['selected'][family]['q'],r['selected'][family]['q'] is None])
                readouts=[('online',a['online'],a['relative_to'],None),('fixed_ema',a['fixed_ema'],a['relative_to'],None)]
                for name,reference in [('matched','JOINT_MATCHED'),('true_pseudo_matched','TRUE_MATCHED')]:
                    if name in a:
                        readouts.append((name,a[name].get('effects'),reference,a[name]['radius']))
                for readout,panels,reference,radius in readouts:
                    if panels is None:
                        writers['B0_EFFECTS'].writerow([dataset,seed,epoch,a['scope'],family,q,None,None,
                            mode,readout,reference,'NOT_INFORMATIVE',radius]+[None]*8)
                        continue
                    for pop,out in panels.items():
                        # The reported effect is branch minus reference. Source U reverses a LOO.
                        sign = -1 if mode=='LOO' else 1
                        source_effect = mode in ('LOO','ONLY') and readout!='true_pseudo_matched'
                        U=[sign*c['population']['delta_auc'] for c in out['cells']]
                        for c,cell in enumerate(out['cells']):
                            extra=([U[c],U[q],(sum(U)-U[q])/5,min(U),sum(U)/6]
                                    if source_effect else [None]*5)
                            writers['B0_EFFECTS'].writerow([dataset,seed,epoch,a['scope'],family,q,pop,c,
                                mode,readout,reference,'COMPLETE',radius,cell['population']['delta_auc'],
                                cell['sampled']['delta_auc'],cell['delta_logloss'],*extra])
                if mode=='LOO':
                    for pop in a['online']:
                        for c in range(6):
                            def D(readout):
                                return a[readout][pop]['cells'][c]['population']['delta_auc']
                            small,fs,fa=D('online_small'),D('F_same_small'),D('fixed_ema')
                            writers['B1_EMA_INTERACTIONS'].writerow([dataset,seed,epoch,a['scope'],family,q,pop,c,
                                D('online'),small,fs,fa,fs-small,D('F_full_delta'),fa-fs])
    for tmp,path in files.values():
        os.replace(tmp,path)
