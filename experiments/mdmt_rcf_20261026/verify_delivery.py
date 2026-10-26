"""One bounded real-input algebra check for the observational deployment compiler."""
from __future__ import annotations
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.special import expit
from .protocol import RoleView,PredictionBundle,RoleBundle
from .prefix import configure,make_native,PilotAccess
from .baselines import DirectHead,fixed_feature_bank,predict_direct_models
from .readers import LinearPredictor
from .formation import DeltaFunction,DeltaMLP,features_from_functions
from .vendor_snapshot.mdmt_dfd_20261026.functions import PilotBank,freeze
from .delivery import FrozenDeployment,_reference_raw,_observation_fingerprint,_native_eval_ready
from .state import digest


def run_required(config):
    configure();data=RoleBundle('movielens')
    # A shape/parity fixture, not a population metric or synthetic research run.
    ids=np.sort(np.concatenate([data.ids('fit',s)[:16] for s in range(3)]))
    raw=torch.from_numpy(np.array(data.x[ids],dtype=np.float32))
    native,_=make_native(data.schema,5201,'movielens','cpu');freeze(native)
    with torch.inference_mode():packet=native.encode(raw)
    h,z=packet.h.numpy(),packet.logits.numpy();domain=data.domain[ids]
    cut=np.asarray([np.median(z[domain==s],axis=0) for s in range(3)])
    groups=np.empty((len(ids),2),np.int64)
    for s in range(3):groups[domain==s]=(z[domain==s]>cut[s]).astype(np.int64)
    view=RoleView(ids,domain,None,h,z,groups)
    bundle=SimpleNamespace(dataset='movielens',seed=5201,views={'fit':view,'Q_explore':view},
        native=native,base_models={},base_logits={},normalizers={'group_cut':cut},
        identity={'verification_only':True},diagnostics={},data=data,device='cpu',
        output_root=Path(config['paths']['artifact'])/'verification_delivery')
    source=freeze(PilotBank());local=[freeze(PilotBank(lambda s,q,k:700+s*10+q)),
                                    freeze(PilotBank(lambda s,q,k:900+s*10+q))]
    with torch.no_grad():
        for bank in [source]+local:
            for head in bank.heads.values():
                head.output.weight.copy_(torch.linspace(-.02,.02,32).reshape(1,-1));head.output.bias.fill_(.01)
    initial=copy.deepcopy(source);initial.train()
    bundle.pilots=PilotAccess(bundle,source,local,initial)
    assert bundle.pilots.initial_bank.training and _native_eval_ready(bundle)
    for t in range(3):
        for q in range(2):
            head=DirectHead(np.zeros(258),np.ones(258),5201+t*10+q,.5)
            with torch.no_grad():head.output.weight.copy_(torch.linspace(-.03,.03,32).reshape(1,-1))
            bundle.base_models[t,q]=(freeze(head),None)
    bundle.base_logits={r:predict_direct_models(bundle,bundle.base_models,r) for r in bundle.views}
    predictions={}
    for method in ('NATIVE','B_REFIT'):
        logits={r:(v.z if method=='NATIVE' else bundle.base_logits[r]) for r,v in bundle.views.items()}
        probabilities={r:expit(v) for r,v in logits.items()}
        predictions[method]=PredictionBundle(method,probabilities,probabilities,{}, {},{})
    for method in ('H_DIRECT','H_DIRECT_EFFORT'):
        models={k:_module[0] for k,_module in bundle.base_models.items()}
        probabilities={r:expit(predict_direct_models(bundle,models,r)) for r in bundle.views}
        predictions[method]=PredictionBundle(method,probabilities,probabilities,{}, {'models':models,'eta':False},{})
    def linear(method,bank,off=False):
        coefs={}
        for t in range(3):
            for q in range(2):
                for g in (0,1):
                    size=len(bank.columns[t,q]);w=np.linspace(-.01,.01,size) if size else np.empty(0)
                    coefs[t,q,g]=dict(alpha=1. if off else 1.1,beta=0. if off else .02,
                                     weights=np.zeros(size) if off else w)
        predictor=LinearPredictor(bundle,bank,coefs,off=off)
        probabilities={r:predictor.predict(r) for r in bundle.views}
        return PredictionBundle(method,probabilities,probabilities,{'selected':'OFF' if off else 'READER'},
            {'selected_coefficients':coefs,'predictor':predictor},{})
    full=fixed_feature_bank(bundle,'FULL_SOURCE_ALL',config)
    predictions['FULL_SOURCE_ALL']=linear('FULL_SOURCE_ALL',full)
    predictions['GROUP_CAL']=linear('GROUP_CAL',fixed_feature_bank(bundle,'GROUP_CAL',config))
    predictions['OFF_CHECK']=linear('OFF_CHECK',full,off=True)
    functions={}
    for s in range(3):
        for q in range(2):
            for k in range(2):
                weight,bias=bundle.pilots.clone_hidden(s,q);model=DeltaMLP(weight,bias)
                with torch.no_grad():
                    model.output.weight.copy_(torch.linspace(-.02,.03,32).reshape(1,-1));model.output.bias.fill_(.01+k*.005)
                fid=f'check_s{s}q{q}k{k}'
                functions[fid]=DeltaFunction(freeze(model),.7,s,q,k,fid)
    formed=features_from_functions(bundle,functions,'RCF',device='cpu')
    predictions['RCF']=linear('RCF',formed)
    before=_observation_fingerprint(bundle,predictions,{})
    errors={};counts={}
    for method,pred in predictions.items():
        deployment=FrozenDeployment(bundle,pred,method)
        actual=deployment(raw);reference=_reference_raw(bundle,pred,method,raw,ids)
        error=float(np.max(np.abs(np.asarray(actual,np.float64)-np.asarray(reference,np.float64))))
        assert error<=1e-6,(method,error)
        errors[method]=error;counts[method]=deployment.parameter_counts()
    after=_observation_fingerprint(bundle,predictions,{})
    assert before==after
    assert counts['OFF_CHECK']['deployed_hidden_parameters']==0
    assert counts['OFF_CHECK']['reader_analytical_parameters']==0
    assert counts['FULL_SOURCE_ALL']['deployed_hidden_parameters']==6*(256*32+32)
    assert counts['OFF_CHECK']['standalone_deployed_parameters']==counts['B_REFIT']['standalone_deployed_parameters']
    return dict(status='PASS',verification_only=True,real_raw_input_rows=len(ids),
        models='untrained canonical native and deterministic unfitted heads; no performance evidence',
        max_probability_errors=errors,probability_tolerance=1e-6,
        off_equals_base_parameter_count=True,off_has_no_source_forward=True,
        native_eval_guard_accepts_training_initial_bank=_native_eval_ready(bundle),
        initial_bank_training_mode_preserved=bundle.pilots.initial_bank.training,
        full_hidden_only_fusion_parameter_count=counts['FULL_SOURCE_ALL']['deployed_hidden_parameters'],
        actual_fused_scalar_count=counts['FULL_SOURCE_ALL']['deployed_fused_linear_scalars'],
        immutable_models_modes_gradients_normalizers_rng=before==after,
        before_hash=digest(before),after_hash=digest(after),benchmark_timing_performed=False,
        model_files_written=0,training_performed=False)


def verify_native_endpoint_reporting(config):
    """Known endpoint-data correction only; no model construction or inference."""
    from .delivery import align_native_terminal_reporting
    from .state import array_hash
    z=np.asarray([[0.,2.],[-3.,.75],[40.,-40.]],dtype=np.float32)
    selected=np.asarray([[.4,.6],[.2,.8],[.3,.7]],dtype=np.float32)
    probability_dict={'validation':selected}
    pred=PredictionBundle('NATIVE',probability_dict,probability_dict,
        {'selection':'fixed selected prefix state'},{},{'verification_only':True})
    bundle=SimpleNamespace(views={'validation':SimpleNamespace(row_id=np.arange(3),domain=np.arange(3))},
        diagnostics={'native_terminal_logits':{'validation':z}},
        identity={'native_epoch24_hash':'verification_known_epoch24_identity',
                  'native_hash':'verification_distinct_selected_identity'})
    selected_hash=array_hash(selected);selection_hash=digest(pred.selection)
    root=Path(config['paths']['artifact'])/'verification_native_endpoint'
    receipt=align_native_terminal_reporting(bundle,{'NATIVE':pred},root)
    assert pred.probabilities is probability_dict
    assert array_hash(pred.probabilities['validation'])==selected_hash
    assert digest(pred.selection)==selection_hash
    assert np.array_equal(pred.terminal_probabilities['validation'],expit(z))
    assert pred.terminal_probabilities['validation'].dtype==np.float32
    with np.load(root/'native_epoch24'/'validation.npz',allow_pickle=False) as companion:
        assert np.array_equal(companion['native_terminal_logits'],z)
        assert np.array_equal(companion['p'],expit(z))
        assert int(companion['epoch'])==24 and 'y' not in companion.files
    second=align_native_terminal_reporting(bundle,{'NATIVE':pred},root)
    assert second['changed_roles']==[]
    return dict(status='PASS',verification_only=True,
        known_native_terminal_logits_dtype=str(z.dtype),selected_probabilities_unchanged=True,
        selected_probability_dictionary_identity_unchanged=True,selection_unchanged=True,
        terminal_equals_exact_fp32_sigmoid_of_recorded_z=True,
        companion_logits_probabilities_epoch_and_identity_valid=True,
        idempotent_second_call=True,models_constructed=0,new_forwards=0,training_performed=False,
        native_replay=False,model_files_written=0,reporting_revision=receipt['revision'])
