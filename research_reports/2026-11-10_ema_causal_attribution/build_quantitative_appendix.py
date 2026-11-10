"""Read-only scalar appendix. No model, guardian, fitting or narrative conclusions."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

SEEDS=tuple(range(2022,2027))
BLOCKS=('movielens__clean','kuairand__clean','movielens__gaussian_d0','kuairand__gaussian_d0')
METRICS=('auc','logloss')
CELLS=tuple((d,t) for d in range(3) for t in range(2))
LATE=('E35','E50')
TERMS=('independent_first','independent_second','deletion_first','deletion_second','interaction','total')
T4=2.7764451052
ROOT=Path(__file__).resolve().parents[2]


def read(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def summary(values):
    clean={str(s):float(values[str(s)]) for s in SEEDS if finite(values.get(str(s)))}
    n=len(clean);mu=mean(clean.values()) if n else None;sd=stdev(clean.values()) if n>1 else None
    half=T4*sd/math.sqrt(5) if n==5 else None
    return dict(n=n,status='COMPLETE' if n==5 else 'INCOMPLETE_SEEDS',mean=mu,sd=sd,
                ci95=[mu-half,mu+half] if half is not None else None,seed_values=values)


class Appendix:
    def __init__(self,out,dest,partial):
        self.out=out;self.dest=dest;self.partial=partial;self.rows=[];self.sources={};self.missing=[];self.tables={}
        self.late_effects={};self.test_effects={};self.endpoint_epochs={}

    def source(self,path):
        key=os.path.relpath(path,ROOT)
        self.sources[key]=dict(sha256=sha(path),link=os.path.relpath(path,self.dest))
        return key

    def table(self,block,name):
        key=(block,name)
        if key not in self.tables:
            directory=self.out/'blocks'/block;index=read(directory/'ARTIFACT_INDEX.json')
            if name not in index:
                self.missing.append(f'{block}/{name}: absent');self.tables[key]=[];return []
            path=directory/Path(index[name]['path']).name;source=self.source(path)
            if self.sources[source]['sha256']!=index[name]['sha256']:raise ValueError('Source hash mismatch: '+source)
            data=read(path)
            if not isinstance(data,list):raise ValueError('Expected scalar table list: '+source)
            self.tables[key]=[dict(r,_source=source,_row=i) for i,r in enumerate(data)]
        return self.tables[key]

    def aggregate(self,section,identity,rows,anchors,metric,effect,reference=None,cells=CELLS):
        """One value per expected seed/anchor/cell, cells first then anchors; no fallback zeros."""
        lookup={}
        for r in rows:
            key=(int(r['seed']),r['anchor'],int(r['domain']),int(r['task']))
            if key in lookup:raise ValueError('Duplicate scalar identity: '+str(identity))
            lookup[key]=r
        values={};provenance=[];epochs={};unavailable=[]
        for seed in SEEDS:
            anchor_values=[];epochs[str(seed)]={}
            for anchor in anchors:
                cell_values=[]
                for d,t in cells:
                    r=lookup.get((seed,anchor,d,t));v=r.get(metric) if r else None
                    ref=reference.get((seed,anchor,d,t)) if reference is not None else None
                    if reference is not None:
                        other=ref.get(metric) if ref else None
                        v=v-other if finite(v) and finite(other) else None
                    if not finite(v):
                        unavailable.append([seed,anchor,d,t]);continue
                    cell_values.append(v)
                    for item in (r,ref):
                        if item is not None:
                            provenance.append([item['_source'],item['_row']])
                    epochs[str(seed)][anchor]=dict(state=r.get('epoch'),reference=ref.get('epoch') if ref else None)
                if len(cell_values)==len(cells):anchor_values.append(mean(cell_values))
            values[str(seed)]=mean(anchor_values) if len(anchor_values)==len(anchors) else None
        item=dict(section=section,identity=dict(identity,metric=metric,effect=effect),anchors=list(anchors),
                  cells=[list(c) for c in cells],aggregation='mean cells within anchor within seed; then mean named anchors within seed',
                  selected_epochs=epochs,source_rows=sorted(set(tuple(p) for p in provenance)),missing_cells=unavailable,**summary(values))
        if not lookup:self.missing.append(str(identity)+': requested contrast not present; no value inferred')
        self.rows.append(item);return item

    @staticmethod
    def lookup(rows):
        result={}
        for r in rows:
            k=(int(r['seed']),r['anchor'],int(r['domain']),int(r['task']))
            if k in result:raise ValueError('Duplicate reference cell')
            result[k]=r
        return result

    def block(self,block):
        data=self.table(block,'INTERVENTION_METRICS')
        def select(name,pop):return [r for r in data if r['descriptor_id']==name and r['population']==pop]
        for stage,anchors in [('early',('E5','E10')),('middle',('E20',)),('late',LATE)]:
            for metric in METRICS:
                for name in ('O','F'):
                    self.aggregate('01_stage_baselines',dict(block=block,population='V',stage=stage,descriptor_id=name),select(name,'V'),anchors,metric,'value')
                item=self.aggregate('01_stage_baselines',dict(block=block,population='V',stage=stage,descriptor_id='F',reference_id='O'),
                    select('F','V'),anchors,metric,'F-O',self.lookup(select('O','V')))
                if stage=='late':self.late_effects[(block,metric)]=item
        for cell in CELLS:
            for metric in METRICS:
                ident=dict(block=block,population='V',stage='late',cell=list(cell))
                self.aggregate('01_late_cells',dict(ident,descriptor_id='O'),select('O','V'),LATE,metric,'value',cells=(cell,))
                self.aggregate('01_late_cells',dict(ident,descriptor_id='F',reference_id='O'),select('F','V'),LATE,metric,'F-O',self.lookup(select('O','V')),cells=(cell,))
        states=['O','B/N','B/S',*[f'B2/{g}/F' for g in ('EMBED','STAR_SKIP','EXPERT')],'C_PRIMARY/recent','C_PRIMARY/old']
        for pop in ('V','V_B'):
            for state in states:
                for metric in METRICS:
                    self.aggregate('02_late_deletions',dict(block=block,population=pop,stage='late',descriptor_id='F',reference_id=state),
                        select('F',pop),LATE,metric,'F-'+state,self.lookup(select(state,pop)))
        functional=self.table(block,'FUNCTIONAL_EFFECTS')
        for contrast in ('O_F','X_F'):
            for cut in ('h','fused'):
                for metric in METRICS:
                    chosen=[r for r in functional if r.get('contrast')==contrast and r.get('cut')==cut and r['population']=='V' and r['metric']==metric]
                    for term in ('independent_first','independent_second','interaction','total'):
                        self.aggregate('03_functional_factorial',dict(block=block,population='V',stage='late',contrast=contrast,cut=cut,contrast_id=f'D/{contrast}/{cut}',outcome_metric=metric),chosen,LATE,term,term)
        temporal=self.table(block,'TEMPORAL_EFFECTS');h=40 if block.startswith('movielens') else 138
        for background in ('O','F'):
            for metric in METRICS:
                chosen=[r for r in temporal if r.get('group')=='EMBED' and r.get('h')==h and r.get('background')==background and r['population']=='V' and r['metric']==metric]
                for term in TERMS:
                    self.aggregate('04_temporal_EMBED',dict(block=block,population='V',stage='late',group='EMBED',h=h,background=background,contrast_id=f'C2/EMBED/{background}',outcome_metric=metric),chosen,LATE,term,term)
        pairs=[r for r in self.table(block,'PAIR_TRANSITIONS') if r.get('descriptor_id')=='F' and r.get('reference_id')=='O' and r['population']=='V']
        for r in pairs:
            if not r.get('pairs'):continue
            count=r['counts'];n=r['pairs']
            corrected=sum(count[i][j]*max((j-i)/2,0) for i in range(3) for j in range(3))/n
            broken=sum(count[i][j]*max((i-j)/2,0) for i in range(3) for j in range(3))/n
            if abs(corrected-r['corrected'])>1e-12 or abs(broken-r['broken'])>1e-12:raise ValueError('Pair normalization mismatch')
        for field in ('corrected','broken','delta_auc'):
            self.aggregate('05_exact_pair',dict(block=block,population='V',stage='late',descriptor_id='F',reference_id='O'),pairs,LATE,field,field)
        ensembles=self.table(block,'ENSEMBLE_METRICS')
        for readout in ('logit','probability'):
            chosen=[r for r in ensembles if r['readout']==readout and r['population']=='V_mech']
            for metric in METRICS:
                self.aggregate('06_ensemble_minus_F',dict(block=block,population='V_mech',stage='late',descriptor_id='q_'+readout,reference_id='F'),chosen,LATE,metric,'q_'+readout+'-F',self.lookup(select('F','V_mech')))
        fusion=self.table(block,'FUSION_EFFECTS')
        for contrast in ('O_F','X_F'):
            for anchor in ('E35','tF'):
                for pop in ('V','V_B'):
                    chosen=[]
                    for r in fusion:
                        if r.get('contrast')==contrast and r.get('component')=='task' and r['anchor']==anchor:
                            for i,v in enumerate(r['metrics'].get(pop,[])):chosen.append(dict(r,domain=i//2,task=i%2,**v))
                    for metric in METRICS:
                        # Removal loss is F minus the task-fusion-deleted state, on the same cells.
                        ref=self.lookup(chosen)
                        self.aggregate('07_task_fusion_deletion',dict(block=block,population=pop,contrast=contrast,component='task',descriptor_id='F',reference_id=f'fusion/{contrast}/task'),
                            select('F',pop),(anchor,),metric,'F-minus-deleted',ref)
        for r in self.table(block,'BRIDGE_SUMMARY'):
            if r['population'] in ('V_mech','V_B') and r['metric'] in ('utility','delta_static','macro_logloss_delta'):
                self.rows.append(dict(section='08_bridge_saved_summary',identity={k:r[k] for k in ('dataset','objective','menu','selector','population','metric')},
                    block=block,aggregation='original saved summary; no refit or recalculation',source_rows=[[r['_source'],r['_row']]],
                    **{k:r[k] for k in ('n','status','mean','sd','ci95','seed_values')}))
        # Selected X and refinement identities are linked without inventing untriggered assays.
        directory=self.out/'blocks'/block
        for name in ('CONTRAST_SELECTION','GROUP_SELECTION','SOURCE_ASSAY_DECISION'):
            path=directory/(name+'.json.gz')
            if path.exists():self.source(path)
        self.tables.clear()

    def test(self):
        path=self.out/'ENDPOINTS.csv';source=self.source(path)
        with path.open() as f:raw=list(csv.DictReader(f))
        expected={'O','F','OatF','S_atF','N_atF'}
        for block in BLOCKS:
            ds,obj=block.split('__');selected=[r for r in raw if r['dataset']==ds and r['objective']==obj]
            if any({r['readout'] for r in selected if int(r['seed'])==s}!=expected for s in SEEDS):raise ValueError('Incomplete five-state Test endpoints: '+block)
            self.endpoint_epochs[block]={str(s):{r['readout']:int(r['epoch']) for r in selected if int(r['seed'])==s} for s in SEEDS}
            records=[]
            for i,r in enumerate(raw):
                if r not in selected:continue
                for d,t in CELLS:
                    records.append(dict(r,seed=int(r['seed']),epoch=int(r['epoch']),anchor='Test',domain=d,task=t,
                        _source=source,_row=i,**{m:float(r[f'd{d}t{t}_{m}']) for m in METRICS}))
            def states(name):return [r for r in records if r['readout']==name]
            views=[('macro',CELLS)]+[(f'd{d}t{t}',((d,t),)) for d,t in CELLS]
            for view,cells in views:
                for metric in METRICS:
                    identity=dict(block=block,population='Test',view=view)
                    for state in sorted(expected):
                        self.aggregate('09_Test_values',dict(identity,descriptor_id=state,identity_kind='state_value'),states(state),('Test',),metric,'value',cells=cells)
                    item=self.aggregate('09_Test_selected',dict(identity,descriptor_id='F',reference_id='O',identity_kind='independent_selected'),states('F'),('Test',),metric,'F-O',self.lookup(states('O')),cells)
                    self.test_effects[(block,metric,view)]=item
                    for state in ('F','S_atF','N_atF'):
                        for ref in ('OatF','F'):
                            if any(self.endpoint_epochs[block][str(s)][state]!=self.endpoint_epochs[block][str(s)][ref] for s in SEEDS):raise ValueError('Same-time Test epoch mismatch')
                            self.aggregate('09_Test_same_time',dict(identity,descriptor_id=state,reference_id=ref,identity_kind='same_time_atF'),states(state),('Test',),metric,state+'-'+ref,self.lookup(states(ref)),cells)

    def interactions(self):
        for dataset in ('movielens','kuairand'):
            for kind,effects in [('late_V',self.late_effects),('Test_selected',self.test_effects)]:
                for metric in METRICS:
                    views=['macro'] if kind=='late_V' else ['macro']+[f'd{d}t{t}' for d,t in CELLS]
                    for view in views:
                        keys=[(dataset+'__'+obj,metric) if kind=='late_V' else (dataset+'__'+obj,metric,view) for obj in ('clean','gaussian_d0')]
                        if any(k not in effects for k in keys):
                            self.missing.append(f'{dataset}/{kind}/{metric}/{view}: Gaussian-Clean interaction unavailable');continue
                        clean,gaussian=[effects[k] for k in keys]
                        values={str(s):gaussian['seed_values'][str(s)]-clean['seed_values'][str(s)]
                            if finite(gaussian['seed_values'][str(s)]) and finite(clean['seed_values'][str(s)]) else None for s in SEEDS}
                        self.rows.append(dict(section='10_Gaussian_minus_Clean',identity=dict(dataset=dataset,population='V' if kind=='late_V' else 'Test',stage=kind,view=view,metric=metric,effect='(F-O)_Gaussian-(F-O)_Clean'),
                            aggregation='paired same-seed difference of the two existing within-seed effects',
                            source_rows=clean['source_rows']+gaussian['source_rows'],**summary(values)))

    def write(self):
        self.dest.mkdir(parents=True,exist_ok=True)
        result=dict(status='PARTIAL' if self.partial else 'COMPLETE_INPUTS',partial=self.partial,
            source_attempt=os.path.relpath(self.out,ROOT),script_sha256=sha(Path(__file__)),seeds=list(SEEDS),
            rules=dict(n_unit='seed',t4=T4,late=list(LATE),tF='separate selected-time anchor, never part of late',
                       pairs='corrected/broken already normalized; never divide twice',missing='never fill zero'),
            missing=self.missing,sources=self.sources,endpoint_epochs=self.endpoint_epochs,rows=self.rows)
        (self.dest/'EVIDENCE_SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        text=['# 定量附录（机械数表）','',
              '**PARTIAL：输入尚不齐全；缺失项目不代表零效应或阴性结果。**' if self.partial else '**四 block 与 Test 输入齐全；本附录不作科学结论。**','',
              '先在 seed 内对指定六格求等权均值，再对指定 anchors 求均值；每格表仅使用该格。n 是 seed 数，n=5 时用 t4=2.7764451052 计算95%CI。桥接行直接保留原 summary。tF 单列，不并入 late。',
              'F−C_PRIMARY/recent（或 old）是 F 减删除态；不是 recent-only（或 old-only）。融合删除也记 F−删除态，LL 保留原符号。','']
        if self.missing:text+=['## 缺失/不可用项目','']+['- '+m for m in self.missing]+['']
        def fmt(v):return '—' if v is None else f'{v:.7g}'
        sections=list(dict.fromkeys(r['section'] for r in self.rows))
        for section in sections:
            text+=['## '+section,'','| 身份/作用方向 | n | 均值 | SD | 95%CI | seed2022→2026原值 | 来源 |','| --- | ---: | ---: | ---: | --- | --- | --- |']
            for r in (v for v in self.rows if v['section']==section):
                ident='; '.join(f'{k}={v}' for k,v in r['identity'].items())
                if r.get('block'):ident=r['block']+'; '+ident
                if r.get('anchors'):ident+='; anchors='+','.join(r['anchors'])
                refs=sorted({s for s,_ in r['source_rows']});links=' '.join(f'[{Path(s).name}]({self.sources[s]["link"]})' for s in refs)
                ci='—' if r.get('ci95') is None else '['+', '.join(fmt(v) for v in r['ci95'])+']'
                vals=', '.join(fmt(r['seed_values'].get(str(s))) for s in SEEDS)
                text.append(f'| {ident.replace("|", "∣")} | {r["n"]} | {fmt(r["mean"])} | {fmt(r["sd"])} | {ci} | {vals} | {links} |')
            text.append('')
        text+=['## Test 选点 epochs','', '未提供 Test。' if not self.endpoint_epochs else '各状态及差值的原/ref epochs 同时保留在 JSON rows.selected_epochs。','']
        for block,seeds in self.endpoint_epochs.items():
            text+=['### '+block,'','| seed | O | F | OatF | S_atF | N_atF |','| --- | --- | --- | --- | --- | --- |']
            for seed,states in seeds.items():text.append('| '+' | '.join([seed]+[str(states[k]) for k in ('O','F','OatF','S_atF','N_atF')])+' |')
            text.append('')
        (self.dest/'QUANTITATIVE_APPENDIX.md').write_text('\n'.join(text)+'\n')
        print(json.dumps(dict(status=result['status'],rows=len(self.rows),sources=len(self.sources),missing=len(self.missing),dest=str(self.dest))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/ema_causal_20261110/attempt2')
    parser.add_argument('--dest',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    out=args.out.resolve();dest=args.dest.resolve()
    if dest==out or out in dest.parents:raise ValueError('Appendix destination must be outside attempt')
    missing=[b for b in BLOCKS if not (out/'blocks'/b/'BLOCK_COMPLETE.json').exists()]
    test_ready=(out/'ENDPOINTS.csv').exists() and (out/'TEST_RECEIPT.json').exists()
    if not args.partial and (missing or not test_ready):raise SystemExit('Complete four-block + Test inputs required; use --partial explicitly')
    a=Appendix(out,dest,args.partial)
    a.missing=[b+': BLOCK_COMPLETE absent' for b in missing]
    for block in BLOCKS:
        if block not in missing:
            receipt=read(out/'blocks'/block/'BLOCK_COMPLETE.json')
            if receipt.get('status')!='COMPLETE':raise ValueError('Block not COMPLETE: '+block)
            a.source(out/'blocks'/block/'BLOCK_COMPLETE.json');a.block(block)
    if test_ready:
        if read(out/'TEST_RECEIPT.json').get('status')!='COMPLETE':raise ValueError('Test receipt not COMPLETE')
        a.source(out/'TEST_RECEIPT.json');a.test()
    else:a.missing.append('Test: ENDPOINTS.csv and COMPLETE TEST_RECEIPT required')
    a.interactions();a.write()


if __name__=='__main__':main()
