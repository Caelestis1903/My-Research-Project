"""Primary-only exact manifests after the preregistered N1 resource gates."""
import argparse
import json
from pathlib import Path
import shutil
import n17_common as c


def manifest(path,tasks,gate_sha):
    c.dump(path,dict(cwd=str(c.ROOT),reservation_mib=1900,tasks=tasks,gate_sha256=gate_sha,source_sha256=c.sha(__file__)))


def entry(name,script,args):
    return dict(name=name,command=[c.PYTHON,'-u','experiments/'+script,*args,'--device','{device}'],log=str(c.OUT/'logs'/(name+'.log')))


def preserve_h100(run):
    receipt=json.loads((run/'receipt.json').read_text());assert receipt['epochs']==100
    records={}
    for name in ('receipt.json','selection_frozen.json','selected_state.pt','trajectory.csv','metrics_selection.csv'):
        src=run/name;dest=run/('H100_'+name)
        if dest.exists():assert c.sha(src)==c.sha(dest),'H100 preservation conflict'
        else:shutil.copyfile(src,dest)
        records[name]=dict(path=str(dest.resolve()),sha256=c.sha(dest))
    c.dump(run/'H100_preservation.json',dict(status='FROZEN',artifacts=records,test_read=False))


def prepare():
    gate_path=c.OUT/'N1_diagnostic_gates.json';gate=json.loads(gate_path.read_text());assert gate['status']=='FROZEN'
    freeze=json.loads((c.OUT/'N1_winners_frozen.json').read_text());assert freeze['status']=='FROZEN';gate_sha=c.sha(gate_path)
    selected=gate['N2_optimizer'];extensions=[];quota=[];geometry=[];local=[];control=[];anchors=[]
    for pair in freeze['pairs']:
        opt=pair['optimizer'];seed=pair['seed'];lr=pair['common_lr']
        for arm in ('SN','SB'):
            run=Path(pair[arm]);receipt=json.loads((run/'receipt.json').read_text());assert receipt['epochs']>=100
            if gate['optimizers'][opt]['extend_H200']:
                assert receipt['epochs']==100,'generate extension only from untouched H100'
                preserve_h100(run)
                name=run.name+'_H200'
                extensions.append(entry(name,'n17_capture.py',['--seed',str(seed),'--arm',arm,'--optimizer',opt,'--lr',str(lr),
                    '--epochs','200','--phase','N1_confirmation','--run-dir',str(run),'--resume']))
            if opt==selected and gate['N2_quota_training']:
                name=f'N2quota_{opt}_lr{lr}_{arm}_seed{seed}'
                quota.append(entry(name,'n17_quota.py',['--seed',str(seed),'--arm',arm,'--optimizer',opt,'--lr',str(lr),
                    '--epochs','100','--phase','N2_quota','--run-dir',str(c.OUT/'runs'/name)]))
            if seed>2035:continue
            for epoch in (10,40):
                checkpoint=run/f'epoch{epoch:03d}.pt';assert checkpoint.exists()
                args=['--checkpoint',str(checkpoint),'--seed',str(seed),'--optimizer',opt,'--lr',str(lr),'--anchor-arm',arm]
                suffix=f'{opt}_{arm}_seed{seed}_E{epoch}'
                if opt==selected:
                    anchors.append(dict(optimizer=opt,seed=seed,anchor_arm=arm,epoch=epoch,checkpoint=str(checkpoint),sha256=c.sha(checkpoint)))
                    name='N2geometry_'+suffix
                    extra=['--adam-map'] if opt=='adam' and gate['optimizers']['adam']['important_residual'] else []
                    geometry.append(entry(name,'n17_geometry.py',[*args,'--out',str(c.OUT/'N2_fixedpoints'/name),'--k','512',*extra]))
                    name='N3development_'+suffix
                    local.append(entry(name,'n17_local.py',[*args,'--out',str(c.OUT/'N3_development'/name),'--draws','32']))
                elif opt=='sgd' and gate['N3_other_SGD_whole_C']:
                    name='N3SGDcontrol_'+suffix
                    control.append(entry(name,'n17_local.py',[*args,'--out',str(c.OUT/'N3_SGD_control'/name),'--draws','32','--only-block','C']))
    assert len(geometry)==len(local)==len(anchors)==16
    for label,tasks in [('H200',extensions),('N2_quota',quota),('N2_fixedpoints',geometry),('N3_development',local),('N3_SGD_control',control)]:
        manifest(c.OUT/(label+'_manifest.json'),tasks,gate_sha)
    c.dump(c.OUT/'N2_N3_anchor_freeze.json',dict(status='FROZEN',anchors=anchors,gate_sha256=gate_sha,test_read=False))
    print(dict(H200=len(extensions),N2_quota=len(quota),N2_fixedpoints=len(geometry),N3_development=len(local),N3_SGD_control=len(control)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();prepare()
