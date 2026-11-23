"""Source, population, parameter support and native-step engineering checks."""
import argparse
import copy
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from . import source

PIN = "f66bf17cba2980618e6f2edd1c7381b73c35cac5"
MAIN = "92a592418d3f11a588f1e628fbe9364ceab71e18"
V3 = Path("experiments/conditional_information_momentum_20261102_v3")
GA31 = V3 / "vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31"
PUBLISHED = Path("m3oe/MDMTRec/models/multi_domain/model.py")
DESIGN = Path("research_reports/mdmt_shared_private_coordination_20261123/DESIGN_SPECIFICATION.md")
ARTIFACTS = source.ROOT / "artifacts/mdmt_shared_private_coordination_20261123/S0"
DATASETS = source.DATASETS


def _git(*args):
    return subprocess.check_output(("git", *map(str, args)), cwd=source.ROOT, text=True).strip()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _source_manifest():
    donor_files = _git("ls-tree", "-r", "--name-only", PIN, V3).splitlines()
    plan_files = _git("ls-tree", "-r", "--name-only", PIN, "research_plans/11-23").splitlines()
    history = Path("experiments/conditional_history_shared_update_20261122")
    receipts = [source.RECEIPTS.relative_to(source.ROOT) / name for name in
                ("A_CORE_SOURCE_MANIFEST.json", *(f"{d}_data_identity.json" for d in DATASETS))]
    provenance = [PUBLISHED, Path("requirements-phase0-3.lock"), *receipts,
                  history / "b_build_panels.py", history / "b_panel_identity.py", history / "state.py"]
    paths = sorted(map(str, [*donor_files, *plan_files, *provenance]))
    main_donor_changed = _git("diff", "--name-only", MAIN, PIN, "--", V3, PUBLISHED)
    if main_donor_changed:
        raise ValueError(f"Fixed1 donor differs from pinned main: {main_donor_changed}")

    pinned = dict(line.split("\t", 1)[::-1] for line in _git(
        "ls-tree", "-r", PIN, *paths).splitlines())
    # Batch working-tree blob verification, including frozen scalar receipts.
    working = _git("hash-object", "--", *paths).splitlines()
    files = {}
    for path, blob in zip(paths, working, strict=True):
        if pinned[path].split()[2] != blob:
            raise ValueError(f"Working source differs from {PIN}: {path}")
        files[path] = dict(git_blob=blob, sha256=source.sha_file(source.ROOT / path))
    if files[str(GA31 / "reference.py")] != dict(
        git_blob="6d1616efa1ce3e028ccd9de4a9b2daae302a184c",
        sha256="f86b9781afb3c227deb8c1c0cdd16450d1bfc90da83bc04ffc30c328540fc538",
    ) or files[str(PUBLISHED)]["git_blob"] != "e2b46e71001b2e0f53681fd5c30872806af4a3db":
        raise ValueError("Unexpected fixed1 or published model source fingerprint")

    design_hash = source.sha_file(source.ROOT / DESIGN)
    if design_hash != "7c1df7c873468de90b17ff53e32deab0721001bdd53dab2600de3b924467a74b":
        raise ValueError("D2 design SHA-256 mismatch")
    configs = {d: source.config(d) for d in DATASETS}
    for dataset, rho, wd in (("movielens", .9828205985452511, 1e-4), ("kuairand", .995, 1e-5)):
        expected = dict(lr=.01, epochs=50, batch_size=40960, rho=rho, weight_decay=wd,
                        arm="A0", lambda_pair=0.0, mode="adam", dynamic=False)
        if {key: configs[dataset][key] for key in expected} != expected:
            raise ValueError(f"{dataset}: clean v3 effective config mismatch")
    return dict(
        status="PASS", source_commit=PIN, matching_main=MAIN, audit_head=_git("rev-parse", "HEAD"),
        design=dict(path=str(DESIGN), sha256=design_hash,
                    git_blob=_git("hash-object", "--", DESIGN)),
        effective_config=configs,
        adam=dict(betas=[.9, .999], eps=1e-8, coupled=True, foreach=False, fused=False,
                  amsgrad=False, capturable=False, differentiable=False, maximize=False),
        model=dict(domains=3, tasks=2, shared_experts=4, fcn=[512, 256, 256, 64],
                   embedding_dim=16, fixed1={name: 1 for name in ("exp_d", "exp_t", "bal_d", "bal_t")}),
        loader=dict(seed_offset=11000, shuffle="one independent CPU randperm per epoch", drop_last=False),
        precision=dict(dtype="float32", amp=False, tf32=False, cudnn_benchmark=False),
        source_layering=dict(
            executable="v3 protocol and objectives A0 with clean PairOptimizer, EMA02, and GA31 fixed reference",
            standalone_ga31="Historical metadata uses global weight_decay 1e-5 and EMA false",
            published="Its trainable fusion weights apply sigmoid to the input value 1",
            preprocessing="GA31 audited_data encoders and array/row hashing with bounded decoders",
        ),
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         torch=str(torch.__version__), cuda_build=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(), numpy=np.__version__, pandas=pd.__version__),
        source_files=files,
    )


def audit(output_dir=ARTIFACTS):
    """Write the checked source and data manifests before numerical checks."""
    output_dir = Path(output_dir)
    _write_json(output_dir / "SOURCE_MANIFEST.json", _source_manifest())
    datasets = {dataset: source.load_data(dataset) for dataset in DATASETS}
    _write_json(output_dir / "DATA_MANIFEST.json", dict(
        status="PASS", datasets={dataset: identity for dataset, (_, identity) in datasets.items()},
        test_status="SEALED",
    ))
    return datasets


def _close(actual, expected, device):
    if device.type == "cpu":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def joint_fidelity(dataset, bundle, device, seed=2022):
    """Compare real first and natural tail batches against donor A0."""
    from experiments.conditional_information_momentum_20261102_v3.objectives import train_step
    from . import blocks

    device = torch.device(device)
    cfg = source.config(dataset)
    source.setup(seed, device)
    model = source.make_model(bundle, dataset).to(device)
    source.setup(seed, device)
    donor = source.v3.make_model(bundle, source.v3.load_protocol()["datasets"][dataset]["source"]).to(device)
    model.train()
    donor.train()
    ema = copy.deepcopy(model).requires_grad_(False)
    donor_ema = copy.deepcopy(donor).requires_grad_(False)
    opt = source.optimizer(model, cfg)
    donor_opt = source.v3.make_optimizer(donor, cfg)
    _close(model.state_dict(), donor.state_dict(), torch.device("cpu"))
    _close(opt.state_dict(), donor_opt.adam.state_dict(), torch.device("cpu"))
    cfg_size = cfg["batch_size"]
    order = torch.randperm(len(bundle.train_y), generator=torch.Generator().manual_seed(seed + 11000)).numpy()
    tail_start = ((len(order) - 1) // cfg_size) * cfg_size
    rows = []
    for position, offset in (("first", 0), ("tail", tail_start)):
        ix = order[offset:offset + cfg_size]
        x = torch.as_tensor(bundle.train_x[ix], device=device).T
        y = torch.as_tensor(bundle.train_y[ix], device=device)
        row_ids = bundle.train_row_ids[ix]
        candidate_pred = []
        handle = model.register_forward_hook(lambda _m, _x, out: candidate_pred.append(out.detach()))
        try:
            candidate = blocks.step(model, opt, ema, ((x, y),), cfg)
        finally:
            handle.remove()
        reference = train_step(
            donor, donor_opt, donor_ema, x, y, row_ids, cfg,
            dict(dataset=dataset, seed=seed, global_step=offset // cfg_size + 1),
        )
        _close(candidate_pred[0], reference["pred"], device)
        _close(candidate["loss"], reference["loss"], device)
        _close(model.state_dict(), donor.state_dict(), device)
        _close(ema.state_dict(), donor_ema.state_dict(), device)
        _close(opt.state_dict(), donor_opt.adam.state_dict(), device)
        rows.append(dict(
            batch=position, rows=len(row_ids), canonical_batch_index=offset // cfg_size + 1,
            row_ids_sha256=source.raw.row_ids_hash(row_ids), loss=candidate["loss"],
            donor_loss=reference["loss"], probability_max_abs_difference=float(
                (candidate_pred[0] - reference["pred"]).abs().max()),
            probability_sha256=source.state_digest(candidate_pred[0]),
            donor_probability_sha256=source.state_digest(reference["pred"]),
            online_sha256=source.state_digest(model.state_dict()),
            donor_online_sha256=source.state_digest(donor.state_dict()),
            ema_sha256=source.state_digest(ema.state_dict()),
            donor_ema_sha256=source.state_digest(donor_ema.state_dict()),
            adam_sha256=source.state_digest(opt.state_dict()),
            donor_adam_sha256=source.state_digest(donor_opt.adam.state_dict()),
            step_summary=candidate,
        ))
    return dict(status="PASS", device=str(device), seed=seed,
                loader_order_sha256=source.raw.row_ids_hash(bundle.train_row_ids[order]),
                comparisons=rows, trajectory="initial state, first joint step, tail joint step")


def _cells_for_parameter(name):
    parts = name.split(".")
    if parts[0] in ("slot_weight", "slot_bias", "domain_expert"):
        return {f"d{parts[1]}t{t}" for t in range(2)}
    if parts[0] == "task_expert":
        return {f"d{d}t{parts[1]}" for d in range(3)}
    if parts[0] in ("gate", "tower"):
        t, d = divmod(int(parts[1]), 3)
        return {f"d{d}t{t}"}
    return {f"d{d}t{t}" for d in range(3) for t in range(2)}


def parameter_audit(dataset, bundle, seed=2022):
    """Compare structural routing, finite perturbations, and six cell gradients."""
    from . import blocks

    source.setup(seed, "cpu")
    model = source.make_model(bundle, dataset)
    cfg = source.config(dataset)
    opt = source.optimizer(model, cfg)
    groups = blocks.groups(model)
    named = dict(model.named_parameters())
    by_id = {id(p): (group, slot) for group, settings in enumerate(opt.param_groups)
             for slot, p in enumerate(settings["params"])}
    if len(by_id) != sum(len(settings["params"]) for settings in opt.param_groups):
        raise ValueError("Repeated native Adam parameter")
    if list(by_id) != [id(p) for p in named.values()]:
        raise ValueError("Native Adam membership or ordering differs from canonical parameters")
    flags = {k: v for k, v in opt.defaults.items() if k != "params"}
    expected_flags = dict(lr=cfg["lr"], betas=(.9, .999), eps=1e-8, weight_decay=cfg["weight_decay"],
                          amsgrad=False, maximize=False, foreach=False, capturable=False,
                          differentiable=False, fused=False, decoupled_weight_decay=False)
    if flags != expected_flags:
        raise ValueError(f"Native Adam numerical flags mismatch: {flags}")

    model_config = dict(domains=model.domain_num, tasks=model.task_num, shared_experts=model.expert_num,
                        fcn=[*model.star_dim[1:], *model.fcn_dim])
    if model_config != dict(domains=3, tasks=2, shared_experts=4, fcn=[512, 256, 256, 64]):
        raise ValueError(f"Unexpected canonical model dimensions: {model_config}")
    fusion = {name: getattr(model, name) for name in ("exp_d", "exp_t", "bal_d", "bal_t")}
    if not all(type(value) in (int, float) and value == 1 for value in fusion.values()):
        raise ValueError("Expected four plain numeric fixed1 fusion constants")
    if any(name.startswith("_weight_") for name in named):
        raise ValueError("Unexpected trained fusion coefficients")

    order = torch.randperm(len(bundle.train_y), generator=torch.Generator().manual_seed(seed + 11000)).numpy()
    domains = bundle.train_x[:, -1]
    indices = np.concatenate([order[domains[order] == d][:64] for d in range(3)])
    x = torch.from_numpy(bundle.train_x[indices]).T
    y = torch.from_numpy(bundle.train_y[indices])
    cells = {f"d{d}t{t}": (x[-1] == d, t) for d in range(3) for t in range(2)}

    gate_inputs = {}
    hooks = [gate.register_forward_pre_hook(
        lambda _module, args, index=index: gate_inputs.__setitem__(index, args[0].requires_grad)
    ) for index, gate in enumerate(model.gate)]
    try:
        probability, logits = source.forward_logits(model, x)
    finally:
        for hook in hooks:
            hook.remove()
    if gate_inputs != {index: False for index in range(6)}:
        raise ValueError("Canonical gates must consume detached representations")
    torch.testing.assert_close(logits.sigmoid(), probability, rtol=1e-6, atol=1e-7)

    gradients = {}
    embedding_rows = {}
    trainable = {name: p for name, p in named.items() if p.requires_grad}
    for cell, (mask, task) in cells.items():
        loss = F.binary_cross_entropy(probability[mask, task], y[mask, task])
        values = torch.autograd.grad(loss, tuple(trainable.values()), allow_unused=True,
                                     retain_graph=cell != next(reversed(cells)))
        gradients[cell] = {}
        embedding_rows[cell] = {}
        for (name, _), grad in zip(trainable.items(), values):
            gradients[cell][name] = ("none" if grad is None else
                                     "nonzero" if torch.count_nonzero(grad).item() else "zero")
            if not name.startswith("embedding.") or grad is None:
                continue
            feature = int(name.split(".")[2])
            active = torch.nonzero(torch.count_nonzero(grad, dim=1), as_tuple=True)[0]
            seen = x[feature, mask].long().unique()
            if not torch.isin(active, seen).all():
                raise ValueError(f"Embedding gradients touched feature rows outside the cell: {name} {cell}")
            embedding_rows[cell][name] = dict(seen=int(seen.numel()), nonzero=int(active.numel()))

    records = []
    summaries = []
    generator = torch.Generator().manual_seed(seed + 12000)
    cell_probs = probability.detach()
    with torch.no_grad():
        for name, parameter in named.items():
            group = next((key for key in blocks.CORE_GROUPS if name in groups[key]), "FROZEN/OTHER")
            slots = by_id.get(id(parameter))
            original = parameter.detach().clone()
            expected = _cells_for_parameter(name) if group != "FROZEN/OTHER" else set()
            probe = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            parameter.add_(probe, alpha=.025)
            try:
                changed = model(x)
            finally:
                parameter.copy_(original)
            actual = {cell for cell, (mask, task) in cells.items()
                      if not torch.equal(changed[mask, task], cell_probs[mask, task])}
            nonzero = {cell for cell in cells if gradients[cell].get(name) == "nonzero"}
            if actual != expected or nonzero - expected:
                raise ValueError(
                    f"{dataset}: support discrepancy for {name}, expected={sorted(expected)}, "
                    f"forward={sorted(actual)}, nonzero_grad={sorted(nonzero)}"
                )
            records.append(dict(
                name=name, group=group, shape=list(parameter.shape), numel=parameter.numel(),
                dtype=str(parameter.dtype), requires_grad=parameter.requires_grad,
                optimizer_member=slots is not None, optimizer_group=slots[0] if slots else None,
                optimizer_slot=slots[1] if slots else None, fixed1_trainable=parameter.requires_grad,
                source_tensor_sha256=source.state_digest(original),
                structural_cells=sorted(expected), forward_cells=sorted(actual),
                gradient_nonzero_cells=sorted(nonzero),
                gradient_zero_cells=[cell for cell in cells if gradients[cell].get(name) == "zero"],
                gradient_none_cells=[cell for cell in cells if gradients[cell].get(name, "none") == "none"],
            ))
            for cell in cells:
                summaries.append(dict(
                    dataset=dataset, seed=seed, arm="S0_SUPPORT_AUDIT", readout="online",
                    epoch="initial", name=name, group=group, cell=cell, structural=cell in expected,
                    forward=cell in actual, gradient=gradients[cell].get(name, "none"),
                    optimizer_member=slots is not None,
                    embedding_rows_seen=embedding_rows[cell].get(name, {}).get("seen"),
                    embedding_rows_nonzero=embedding_rows[cell].get(name, {}).get("nonzero"),
                ))
    buffers = [dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                    source_tensor_sha256=source.state_digest(value), group="FROZEN/OTHER",
                    optimizer_member=False) for name, value in model.named_buffers()]
    return dict(
        status="PASS", model_class=f"{type(model).__module__}.{type(model).__name__}", model=model_config,
        parameters=records, buffers=buffers, fusion_constants=fusion, native_adam=flags,
        frozen_group="FROZEN/OTHER", group_counts={name: len(members) for name, members in groups.items()},
        gate_detached_inputs=sorted(gate_inputs),
        audited_row_ids_sha256=source.raw.row_ids_hash(bundle.train_row_ids[indices]),
        audited_rows=len(indices), logits_max_abs_probability_difference=float(
            (logits.detach().sigmoid() - cell_probs).abs().max()),
        routing=dict(native=["d0t0", "d1t0", "d2t0", "d0t1", "d1t1", "d2t1"],
                     report=list(cells)),
    ), summaries


def cpu_checks(output_dir=ARTIFACTS, datasets=None):
    """Audit source support and canonical joint updates on the actual CPU batches."""
    output_dir = Path(output_dir)
    datasets = datasets if datasets is not None else audit(output_dir)
    manifest_hash = source.sha_file(output_dir / "SOURCE_MANIFEST.json")
    parameters, fidelity, support = {}, {}, []
    for dataset, (bundle, identity) in datasets.items():
        config_hash = source.state_digest(sorted(source.config(dataset).items()))
        audit_result, rows = parameter_audit(dataset, bundle)
        parameters[dataset] = audit_result
        print(f"{dataset} parameter support passed", flush=True)
        support.extend(dict(row, source_hash=manifest_hash, data_hash=identity["data_hash"],
                            config_hash=config_hash) for row in rows)
        fidelity[dataset] = joint_fidelity(dataset, bundle, "cpu")
        print(f"{dataset} first/tail native Adam fidelity passed", flush=True)
    _write_json(output_dir / "PARAMETER_MANIFEST.json", dict(status="PASS", datasets=parameters))
    _write_json(output_dir / "MODEL_FIDELITY.json", dict(status="PASS", datasets=fidelity))
    with (output_dir / "PARAMETER_SUPPORT.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(support[0]))
        writer.writeheader()
        writer.writerows(support)
    _write_json(output_dir / "CPU_TESTS.json", dict(
        status="PASS", datasets=list(datasets), fidelity="MODEL_FIDELITY.json",
        parameter_support="PARAMETER_SUPPORT.csv",
    ))
    return fidelity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "cpu"):
        stage = subcommands.add_parser(command)
        stage.add_argument("--output", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        audit(args.output)
    else:
        cpu_checks(args.output)


if __name__ == "__main__":
    main()
