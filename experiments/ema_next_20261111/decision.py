"""Predeclared five-seed gates; all inputs are committed Validation records."""
import math
from .protocol import SEEDS,CELLS

T4=2.7764451052


def statistics(values):
    values=[float(x) for x in values];n=len(values);mean=sum(values)/n
    sd=math.sqrt(sum((x-mean)**2 for x in values)/(n-1)) if n>1 else 0.
    half=T4*sd/math.sqrt(n)
    return dict(values=values,n=n,mean=mean,sd=sd,ci95=[mean-half,mean+half],positive_seeds=sum(x>0 for x in values))


def metrics(point):return point.get('metrics',point.get('validation',point))
def auc(point):return metrics(point)['macro_auc']


def selected(records,rid):
    eligible=[r for r in records if rid in r['readouts']]
    if not eligible:raise ValueError('Missing readout '+rid)
    row=max(eligible,key=lambda r:(auc(r['readouts'][rid]),-r['epoch']))
    return dict(row['readouts'][rid],epoch=row['epoch'],job_id=row['job_id'])


def paired(candidate,control):
    if set(candidate)!=set(SEEDS) or set(control)!=set(SEEDS):return dict(status='WAITING_FOR_FIVE_SEEDS')
    cells={c:statistics([metrics(candidate[s])[c+'_auc']-metrics(control[s])[c+'_auc'] for s in SEEDS]) for c in CELLS}
    ll=statistics([metrics(candidate[s])['macro_logloss']-metrics(control[s])['macro_logloss'] for s in SEEDS])
    return dict(status='COMPLETE',auc=statistics([auc(candidate[s])-auc(control[s]) for s in SEEDS]),logloss=ll,cells=cells,
        cell_guard=all(v['mean']>=-.001 for v in cells.values()),candidate_U=sum(auc(candidate[s]) for s in SEEDS)/5,
        control_U=sum(auc(control[s]) for s in SEEDS)/5,
        record_ids=[dict(seed=s,candidate=dict(job_id=candidate[s].get('job_id'),epoch=candidate[s].get('epoch'),state_id=candidate[s].get('state_id')),
                         control=dict(job_id=control[s].get('job_id'),epoch=control[s].get('epoch'),state_id=control[s].get('state_id'))) for s in SEEDS])


def value_gate(name,candidate,control,minimum=.0002):
    p=paired(candidate,control)
    if p['status']!='COMPLETE':return dict(gate=name,status=p['status'],passed=False)
    passed=p['auc']['mean']>=minimum and p['cell_guard']
    return dict(gate=name,status='ACCEPT' if passed else 'STOP',passed=passed,minimum_auc=minimum,evidence=p,
        logloss_decision='IMPROVED' if p['logloss']['mean']<0 else 'LL_COST_OR_NO_IMPROVEMENT')


def choose(options,control=None,tolerance=1e-6,order=None):
    valid=[]
    for rid,points in options.items():
        if set(points)!=set(SEEDS):continue
        if control is not None and not paired(points,control)['cell_guard']:continue
        valid.append(rid)
    if not valid:return None
    means={k:sum(auc(v) for v in options[k].values())/5 for k in valid};best=max(means.values())
    tied=[k for k in valid if best-means[k]<=tolerance]
    rank={x:i for i,x in enumerate(order or sorted(tied))}
    return min(tied,key=lambda x:(rank.get(x,len(rank)),x))


def complete_curves(parents,required_rids):
    return set(parents)==set(SEEDS) and all({r['epoch'] for r in parents[s]}==set(range(1,51)) and
        all(set(required_rids)<=set(r['readouts']) for r in parents[s]) for s in SEEDS)


def parent_decisions(parents,menu,history_base):
    rids=[v['id'] for v in menu]
    if not complete_curves(parents,rids):return dict(status='WAITING_FOR_FIVE_SEEDS')
    points={rid:{s:selected(parents[s],rid) for s in SEEDS} for rid in rids};F=points['F'];gates={}
    B=[r for r in menu if r['family']=='B_SPLIT']
    windows=sorted({r['H'] for r in B});window_order=[history_base,*[h for h in windows if h!=history_base]]
    order=[r['id'] for r in sorted(B,key=lambda r:(r['scope']!='S',r['mode']!='logit',window_order.index(r['H'])))]
    best=choose({r['id']:points[r['id']] for r in B},F,order=order)
    if best is None:best=choose({r['id']:points[r['id']] for r in B},order=order)
    gates['G_B']=value_gate('G_B',points[best],F);gates['G_B_SHARED']=value_gate('G_B_SHARED',points['B_SHARED'],F)
    for mode in ('PRED','PARAM'):gates['G_C_GLOBAL_'+mode]=value_gate('G_C_GLOBAL',points['C_GLOBAL_'+mode],F)
    unified=['F','B_SHARED',best,'C_GLOBAL_PRED','C_GLOBAL_PARAM'];u_best=choose({k:points[k] for k in unified})
    for mode in ('PRED','PARAM'):
        global_id='C_GLOBAL_'+mode
        for shrink in (.5,1.):
            rid=f'C_CELL_{mode}_s{shrink:g}';aggregates={k:[] for k in ('alpha_abs_difference','pair_rank_change_rate','pb_risk_delta','pb_auc_delta','v_auc_delta')};cell_delta={c:[] for c in CELLS};complete=True
            for seed in SEEDS:
                future=[r for r in parents[seed] if 21<=r['epoch']<=50];values={k:[] for k in aggregates};cells={c:[] for c in CELLS}
                for r in future:
                    diag=r.get('diagnostics',{}).get('C',{}).get(rid,{})
                    if not diag.get('complete',False):complete=False;continue
                    for k in ('alpha_abs_difference','pair_rank_change_rate','pb_risk_delta','pb_auc_delta'):values[k].append(diag[k])
                    values['v_auc_delta'].append(auc(r['readouts'][rid])-auc(r['readouts'][global_id]))
                    for c in CELLS:cells[c].append(metrics(r['readouts'][rid])[c+'_auc']-metrics(r['readouts'][global_id])[c+'_auc'])
                if any(len(v)!=30 for v in values.values()):complete=False
                if complete:
                    for k,v in values.items():aggregates[k].append(sum(v)/30)
                    for c,v in cells.items():cell_delta[c].append(sum(v)/30)
            if not complete:
                mech=dict(gate='G_C_MECH',status='BLOCKED_PANEL',passed=False,readout=rid)
            else:
                stats={k:statistics(v) for k,v in aggregates.items()};cell_stats={c:statistics(v) for c,v in cell_delta.items()}
                passed=(stats['alpha_abs_difference']['mean']>1e-6 and stats['pair_rank_change_rate']['mean']>0 and stats['pb_risk_delta']['mean']<0 and stats['pb_auc_delta']['mean']>0 and stats['v_auc_delta']['mean']>=.0001 and all(v['mean']>=-.001 for v in cell_stats.values()))
                reason='ACCEPT' if passed else 'NO_ACTION_CHANGE' if stats['alpha_abs_difference']['mean']<=1e-6 or stats['pair_rank_change_rate']['mean']<=0 else 'NO_FUTURE_SIGNAL' if stats['pb_risk_delta']['mean']>=0 or stats['pb_auc_delta']['mean']<=0 else 'NO_INCREMENTAL_VALUE'
                mech=dict(gate='G_C_MECH',status='ACCEPT' if passed else 'STOP',passed=passed,reason=reason,readout=rid,evidence=stats,cells=cell_stats)
            own=value_gate('G_C_SELECTED_GLOBAL',points[rid],points[global_id]);unified_gate=value_gate('G_C_SELECTED_UNIFIED',points[rid],points[u_best])
            gates['G_C_MECH_'+rid]=mech
            gates['G_C_SELECTED_'+rid]=dict(gate='G_C_SELECTED',passed=mech['passed'] and own['passed'] and unified_gate['passed'],
                status='ACCEPT' if mech['passed'] and own['passed'] and unified_gate['passed'] else 'STOP',same_mode=own,best_unified=unified_gate,unified_id=u_best)
    accepted=[]
    if gates['G_B']['passed']:accepted.append(best)
    if gates['G_B_SHARED']['passed']:accepted.append('B_SHARED')
    for mode in ('PRED','PARAM'):
        if gates['G_C_GLOBAL_'+mode]['passed']:accepted.append('C_GLOBAL_'+mode)
        choices={f'C_CELL_{mode}_s{s:g}':points[f'C_CELL_{mode}_s{s:g}'] for s in (.5,1.) if gates['G_C_SELECTED_'+f'C_CELL_{mode}_s{s:g}']['passed']}
        winner=choose(choices,F,order=[f'C_CELL_{mode}_s0.5',f'C_CELL_{mode}_s1'])
        if winner:accepted.append(winner)
    return dict(status='COMPLETE',B_winner=best,B_spec=next(r for r in menu if r['id']==best),gates=gates,accepted=accepted,
                points=points,best_unified=u_best)


def short_gate(parents,shorts,family,alpha=None):
    # shorts[(seed,anchor,alpha)] contains exactly the six real feedback epochs.
    required=[(s,a,alpha) for s in SEEDS for a in (20,35)]
    if any(k not in shorts or {r['epoch'] for r in shorts[k]}!=set(range(k[1]+1,k[1]+7)) for k in required):
        return dict(gate='G_A' if family=='A_MATCH' else 'G_A_RAW',status='WAITING_FOR_FIVE_SEEDS',passed=False)
    tails={};online=[];cell_values={c:[] for c in CELLS};e20=[]
    for seed in SEEDS:
        parent={r['epoch']:r for r in parents[seed]};fd=[];od=[];cs={c:[] for c in CELLS}
        for anchor in (20,35):
            rows={r['epoch']:r for r in shorts[(seed,anchor,alpha)]};tail=[anchor+5,anchor+6]
            values=[auc(rows[e]['readouts']['F'])-auc(parent[e]['readouts']['F']) for e in tail]
            fd.append(sum(values)/2)
            od.append(sum(auc(rows[e]['readouts']['O'])-auc(parent[e]['readouts']['O']) for e in tail)/2)
            for c in CELLS:cs[c].append(sum(metrics(rows[e]['readouts']['F'])[c+'_auc']-metrics(parent[e]['readouts']['F'])[c+'_auc'] for e in tail)/2)
        tails[seed]=sum(fd)/2;e20.append(fd[0]);online.append(sum(od)/2)
        for c in CELLS:cell_values[c].append(sum(cs[c])/2)
    f=statistics(tails.values());o=statistics(online);early=statistics(e20);cells={c:statistics(v) for c,v in cell_values.items()}
    passed=f['mean']>=.0001 and early['mean']>0 and all(v['mean']>=-.001 for v in cells.values())
    reason='ACCEPT' if passed else 'REVISE_TIMING' if f['mean']>=.0001 and early['mean']<=0 and all(v['mean']>=-.001 for v in cells.values()) else 'F_COST' if f['mean']<=-.0001 else 'ONLINE_GAIN_WITHOUT_EXTRA_F_GAIN' if o['mean']>=.0001 and f['mean']<.0001 else 'LOCAL_GAIN_NOT_SUSTAINED'
    return dict(gate='G_A' if family=='A_MATCH' else 'G_A_RAW',status='ACCEPT' if passed else 'REVISE' if reason=='REVISE_TIMING' else 'STOP',
        passed=passed,direction_alpha=alpha,F=f,online=o,E20=early,cells=cells,reason=reason)


def match_decision(parents,shorts):
    if len(shorts)!=20:return dict(status='WAITING_FOR_FIVE_SEEDS')
    gates={str(a):short_gate(parents,shorts,'A_MATCH',a) for a in (.5,1.)}
    passing=[a for a in (.5,1.) if gates[str(a)]['passed']]
    chosen=None
    if passing:
        best=max(gates[str(a)]['F']['mean'] for a in passing)
        chosen=min(a for a in passing if best-gates[str(a)]['F']['mean']<=1e-6)
    return dict(status='COMPLETE',gates=gates,winner=chosen)


def teacher_choice(points,accepted,menu):
    specs={v['id']:v for v in menu};pool=[k for k in accepted if specs.get(k,{}).get('family') in ('B_SPLIT','C_GLOBAL_PRED','C_CELL_PRED')]
    single=[k for k in points if k in ('F','B_SHARED','C_GLOBAL_PARAM') or (k.startswith('A_FULL_') and k.endswith('_F') and k in accepted)]
    one=choose({k:points[k] for k in single});order=sorted(pool,key=lambda k:({'B_SPLIT':0,'C_GLOBAL_PRED':1,'C_CELL_PRED':2}[specs[k]['family']],k))
    teacher=choose({k:points[k] for k in pool},order=order)
    gate=value_gate('G_KD_ENTER',points[teacher],points[one]) if teacher else dict(gate='G_KD_ENTER',passed=False,status='NOT_TRIGGERED',reason='NO_QUALIFIED_TWO_STATE_TEACHER')
    return dict(teacher=teacher if gate['passed'] else None,best_one_forward=one,gate=gate)


def ab_gate(ab,a,b):
    ga=value_gate('G_AB_A',ab,a);gb=value_gate('G_AB_B',ab,b)
    return dict(gate='G_AB',passed=ga['passed'] and gb['passed'],status='ACCEPT' if ga['passed'] and gb['passed'] else 'STOP',vs_A=ga,vs_B=gb)


def kd_history_gate(student,parent,label,fteacher,teacher):
    gates=[value_gate('G_KD_HISTORY_'+name,student,control) for name,control in [('F',parent),('LABEL',label),('FTEACHER',fteacher)]]
    gain=paired(teacher,parent)['auc']['mean'];retention=paired(student,parent)['auc']['mean']/gain if gain>0 else None
    passed=all(g['passed'] for g in gates) and retention is not None and retention>=.5
    return dict(gate='G_KD_HISTORY',passed=passed,status='ACCEPT' if passed else 'STOP',comparisons=gates,retention=retention)


def primary_choice(points,accepted,specs):
    if not accepted:return None
    highest=max(sum(auc(x) for x in points[k].values())/5 for k in accepted)
    tied=[k for k in accepted if highest-sum(auc(x) for x in points[k].values())/5<=1e-4]
    return min(tied,key=lambda k:(specs[k].get('forwards',1),specs[k].get('parameter_bytes',0),specs[k].get('extra_steps',0),k))
