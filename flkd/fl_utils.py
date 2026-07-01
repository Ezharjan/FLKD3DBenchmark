"""Federated-learning utilities for the FL+KD 3D point-cloud classification benchmark.

Client-side update routines, server-side aggregators, and client data
partitioning helpers (IID, extreme label-skew, Dirichlet).

Methods (13 algorithms; ``vanilla`` is a CLI alias of FedAvg)
------------------------------------------------------------
FedAvg, FedProx, SCAFFOLD, FedDyn, FedAvgM, FedAdam, FedYogi, FedAdagrad,
FedMedian, FedBN, MOON, Ditto, FedNova.
All client updates support bf16 autocast (``amp=True``) on bf16-capable GPUs;
GPUs without native bf16 run exact fp32.
"""

import copy
import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from flkd.engine import augment_points as _augment_points, autocast_ctx  # noqa: E402


# ---------------------------------------------------------------------------
# Client partitioning
# ---------------------------------------------------------------------------

def _gather_targets(dataset):
    """Collect integer labels from a dataset, using fast paths where possible."""
    labels = getattr(dataset, 'labels', None)
    if labels is not None:
        return np.asarray([int(x) for x in labels], dtype=np.int64)
    # ModelNetDataLoader exposes datapath + classes (avoid per-item disk reads).
    if hasattr(dataset, 'datapath') and hasattr(dataset, 'classes'):
        return np.asarray([int(dataset.classes[dp[0]]) for dp in dataset.datapath], dtype=np.int64)
    if isinstance(dataset, torch.utils.data.Subset):
        base = _gather_targets(dataset.dataset)
        return base[np.asarray(dataset.indices, dtype=np.int64)]
    targets = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        if isinstance(target, torch.Tensor):
            target = int(target.item()) if target.ndim == 0 else int(target.view(-1)[0].item())
        targets.append(int(target))
    return np.asarray(targets, dtype=np.int64)


def split_dataset(dataset, num_clients, iid=False, partition='label_skew',
                  dirichlet_alpha=0.5, seed=None):
    """Split a dataset across clients.

    ``partition`` is one of ``{'iid', 'label_skew', 'dirichlet'}``.
    ``label_skew`` sorts samples by label before splitting (extreme non-IID).
    ``dirichlet`` produces the standard Dirichlet(alpha) heterogeneous
    partition (smaller alpha -> more skew).  ``iid=True`` overrides ``partition``.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    mode = (partition or 'label_skew').lower()
    if iid:
        mode = 'iid'

    n = len(dataset)
    if n == 0:
        raise ValueError('Empty dataset cannot be partitioned')

    if mode == 'iid':
        indices = np.arange(n)
        rng.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)
    elif mode in {'label_skew', 'sort', 'sorted'}:
        targets = _gather_targets(dataset)
        sorted_indices = np.argsort(targets, kind='stable')
        client_indices = np.array_split(sorted_indices, num_clients)
    elif mode in {'dirichlet', 'dir'}:
        targets = _gather_targets(dataset)
        num_classes = int(targets.max()) + 1
        for _ in range(50):
            buckets = [[] for _ in range(num_clients)]
            for cls in range(num_classes):
                idx_c = np.where(targets == cls)[0]
                if idx_c.size == 0:
                    continue
                rng.shuffle(idx_c)
                props = rng.dirichlet([dirichlet_alpha] * num_clients)
                cuts = (np.cumsum(props) * idx_c.size).astype(int)[:-1]
                for cid, chunk in enumerate(np.split(idx_c, cuts)):
                    buckets[cid].extend(chunk.tolist())
            if all(len(b) > 0 for b in buckets):
                break
        client_indices = [np.array(b, dtype=np.int64) for b in buckets]
    else:
        raise ValueError(f'Unknown partition mode: {partition}')

    return [torch.utils.data.Subset(dataset, list(map(int, idx))) for idx in client_indices]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def state_dict_size_mb(model):
    """Total size (MB) of all parameters + buffers."""
    p = sum(t.nelement() * t.element_size() for t in model.parameters())
    b = sum(t.nelement() * t.element_size() for t in model.buffers())
    return (p + b) / (1024 ** 2)


def per_round_communication_mb(model, num_clients, bidirectional=True):
    base = state_dict_size_mb(model) * num_clients
    return base * (2.0 if bidirectional else 1.0)


def initialize_state_like(model, fill_value=0.0):
    return {name: torch.full_like(p.data, fill_value) for name, p in model.named_parameters()}


def initialize_scaffold_control_variates(model):
    return {name: torch.zeros_like(p.data) for name, p in model.named_parameters()}


def aggregate_scaffold_control_variates(client_c_locals, client_weights=None):
    if client_weights is None:
        client_weights = [1.0 / len(client_c_locals)] * len(client_c_locals)
    else:
        total = float(sum(client_weights))
        client_weights = [w / total for w in client_weights]
    c_global = {name: torch.zeros_like(v) for name, v in client_c_locals[0].items()}
    for name in c_global:
        for w, c in zip(client_weights, client_c_locals):
            c_global[name] = c_global[name] + w * c[name]
    return c_global


def count_local_steps(data_loader, local_epochs):
    """Number of optimizer steps a client performs each round."""
    try:
        n_batches = len(data_loader)
    except TypeError:
        n_batches = int(data_loader)
    return max(1, local_epochs * max(1, n_batches))


# ---------------------------------------------------------------------------
# Aggregation routines
# ---------------------------------------------------------------------------

def fedavg_aggregate(client_models, client_weights=None):
    """Weighted FedAvg (McMahan et al., 2017)."""
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    else:
        total = float(sum(client_weights))
        client_weights = [w / total for w in client_weights]

    global_model = copy.deepcopy(client_models[0])
    global_dict = global_model.state_dict()
    for k in global_dict:
        if global_dict[k].dtype in (torch.long, torch.int64, torch.int32):
            stacked = torch.stack([m.state_dict()[k] for m in client_models])
            global_dict[k] = torch.mode(stacked, dim=0).values
        else:
            acc = torch.zeros_like(global_dict[k], dtype=torch.float32)
            for w, m in zip(client_weights, client_models):
                acc = acc + w * m.state_dict()[k].to(torch.float32)
            global_dict[k] = acc.to(global_dict[k].dtype)
    global_model.load_state_dict(global_dict)
    return global_model


def fedmedian_aggregate(client_models):
    """Element-wise coordinate median (Yin et al., 2018)."""
    global_model = copy.deepcopy(client_models[0])
    global_dict = global_model.state_dict()
    for k in global_dict:
        stacked = torch.stack([m.state_dict()[k] for m in client_models])
        if stacked.dtype in (torch.long, torch.int32, torch.int64):
            global_dict[k] = torch.mode(stacked, dim=0).values
        else:
            global_dict[k] = torch.median(stacked.to(torch.float32), dim=0).values.to(global_dict[k].dtype)
    global_model.load_state_dict(global_dict)
    return global_model


def fedavgm_update(global_model, aggregated_model, velocity, server_lr=1.0, server_momentum=0.9):
    """Server-side SGD with momentum (Hsu et al., 2019)."""
    new_global = copy.deepcopy(global_model)
    g = global_model.state_dict()
    a = aggregated_model.state_dict()
    new_vel = {}
    for name in g:
        if g[name].dtype in (torch.long, torch.int64, torch.int32, torch.bool):
            g[name] = a[name]
            continue
        delta = (g[name] - a[name]).to(torch.float32)
        vel_prev = velocity.get(name, torch.zeros_like(delta))
        vel_new = server_momentum * vel_prev + delta
        new_vel[name] = vel_new
        g[name] = (g[name].to(torch.float32) - server_lr * vel_new).to(g[name].dtype)
    new_global.load_state_dict(g)
    return new_global, new_vel


def _is_float_tensor(t):
    return t.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def fedopt_update(global_model, aggregated_model, m_state, v_state, step, mode='adam',
                  server_lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-3):
    """Adaptive federated optimisation: FedAdam / FedYogi / FedAdagrad.

    Reddi et al. (ICLR 2021), Algorithm 2.  The server treats the average client
    delta ``Delta = aggregated - global`` as a pseudo-gradient:

        m_t = beta1*m + (1-beta1)*Delta
        v_t = v + Delta^2                                   (FedAdagrad)
            = v - (1-beta2)*Delta^2*sign(v - Delta^2)       (FedYogi)
            = beta2*v + (1-beta2)*Delta^2                   (FedAdam)
        x_t = x + server_lr * m_t / (sqrt(v_t) + eps)

    Integer/boolean buffers (e.g. ``num_batches_tracked``) are copied straight
    from the aggregated model rather than receiving an adaptive update.
    """
    assert mode in {'adam', 'yogi', 'adagrad'}
    new_global = copy.deepcopy(global_model)
    g = global_model.state_dict()
    a = aggregated_model.state_dict()
    out = new_global.state_dict()
    new_m, new_v = {}, {}
    step = step + 1
    for name in g:
        if not _is_float_tensor(g[name]):
            out[name] = a[name]
            continue
        gf = g[name].to(torch.float32)
        delta = a[name].to(torch.float32) - gf
        mp = m_state.get(name, torch.zeros_like(gf))
        vp = v_state.get(name, torch.zeros_like(gf))
        mc = beta1 * mp + (1 - beta1) * delta
        d2 = delta * delta
        if mode == 'adam':
            vc = beta2 * vp + (1 - beta2) * d2
        elif mode == 'yogi':
            vc = vp - (1 - beta2) * d2 * torch.sign(vp - d2)
        else:  # adagrad
            vc = vp + d2
        update = gf + server_lr * mc / (torch.sqrt(vc) + eps)
        out[name] = update.to(g[name].dtype)
        new_m[name] = mc
        new_v[name] = vc
    new_global.load_state_dict(out)
    return new_global, new_m, new_v, step


def fedadam_update(global_model, aggregated_model, m_state, v_state, step,
                   server_lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-3):
    """Server-side Adam (thin wrapper over :func:`fedopt_update`)."""
    return fedopt_update(global_model, aggregated_model, m_state, v_state, step,
                         mode='adam', server_lr=server_lr, beta1=beta1, beta2=beta2, eps=eps)


# ---------------------------------------------------------------------------
# FedBN helpers (Li et al., 2021): keep BatchNorm layers client-local
# ---------------------------------------------------------------------------

def bn_param_names(model):
    """Return the set of ``state_dict`` keys that belong to normalisation layers."""
    norm_types = (torch.nn.modules.batchnorm._BatchNorm,
                  torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.InstanceNorm1d)
    bn_keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            prefix = f'{module_name}.' if module_name else ''
            for pname, _ in module.named_parameters(recurse=False):
                bn_keys.add(prefix + pname)
            for bname, _ in module.named_buffers(recurse=False):
                bn_keys.add(prefix + bname)
    return bn_keys


def broadcast_non_bn(global_model, client_models, bn_keys):
    """Copy non-BN parameters from ``global_model`` into each client, leaving
    each client's BatchNorm parameters/buffers untouched (FedBN behaviour)."""
    g = global_model.state_dict()
    for cm in client_models:
        cd = cm.state_dict()
        for k in g:
            if k not in bn_keys:
                cd[k] = g[k].clone()
        cm.load_state_dict(cd)


def fednova_aggregate(global_model, client_models, client_taus, client_weights=None):
    """FedNova: aggregate normalised local updates (Wang et al., 2020).

    new_global = global - tau_eff * sum_i p_i * (global - local_i) / tau_i
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    else:
        total = float(sum(client_weights))
        client_weights = [w / total for w in client_weights]
    tau_eff = float(sum(p * t for p, t in zip(client_weights, client_taus)))

    new_global = copy.deepcopy(global_model)
    g = global_model.state_dict()
    new_dict = new_global.state_dict()
    client_dicts = [m.state_dict() for m in client_models]
    for k in g:
        if g[k].dtype in (torch.long, torch.int64, torch.int32):
            stacked = torch.stack([cd[k] for cd in client_dicts])
            new_dict[k] = torch.mode(stacked, dim=0).values
            continue
        delta = torch.zeros_like(g[k], dtype=torch.float32)
        for w, t, cd in zip(client_weights, client_taus, client_dicts):
            if t <= 0:
                continue
            delta = delta + w * (g[k].to(torch.float32) - cd[k].to(torch.float32)) / float(t)
        new_dict[k] = (g[k].to(torch.float32) - tau_eff * delta).to(g[k].dtype)
    new_global.load_state_dict(new_dict)
    return new_global


# ---------------------------------------------------------------------------
# Client-side updates
# ---------------------------------------------------------------------------

def fedprox_client_update(model, global_model, optimizer, data_loader,
                          criterion, device, local_epochs, mu=0.01, amp=False):
    """Local SGD with the FedProx proximal regulariser (Li et al., 2020)."""
    model.train()
    global_model.eval()
    g_params = {name: p.data.clone() for name, p in global_model.named_parameters()}
    for _ in range(local_epochs):
        for _, (points, target) in enumerate(data_loader):
            if points.size(0) <= 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            points = points.to(device, non_blocking=True)  # augmented in the loader worker
            target = target.to(device).long()
            with autocast_ctx(device, amp):
                pred, trans_feat = model(points)
                loss = criterion(pred, target, trans_feat)
            prox = 0.0
            for name, p in model.named_parameters():
                prox = prox + torch.sum((p - g_params[name]) ** 2)
            loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
    return model


def scaffold_client_update(model, global_model, optimizer, data_loader, criterion,
                           device, local_epochs, c_global, c_local, learning_rate,
                           amp=False):
    """SCAFFOLD client update with control-variate correction (Karimireddy et al., 2020).

    Uses SCAFFOLD option-II for the control variate:
        c_i^+ = c_i - c + (x_global - x_local) / (K * eta_l)
    where ``K`` is the number of local SGD steps taken (batches x epochs).  Local
    optimisation is plain SGD with the control-variate correction, so the
    externally supplied ``optimizer`` is not stepped.
    """
    _ = global_model, optimizer
    model.train()
    init_params = {name: p.data.clone() for name, p in model.named_parameters()}
    cg = {name: t.to(device) for name, t in c_global.items()}
    cl = {name: t.to(device) for name, t in c_local.items()}
    local_steps = 0
    for _ in range(local_epochs):
        for _, (points, target) in enumerate(data_loader):
            if points.size(0) <= 1:
                continue
            points = points.to(device, non_blocking=True)  # augmented in the loader worker
            target = target.to(device).long()
            model.zero_grad(set_to_none=True)
            with autocast_ctx(device, amp):
                pred, trans_feat = model(points)
                loss = criterion(pred, target, trans_feat)
            loss.backward()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    p.data = p.data - learning_rate * (p.grad + cg[name] - cl[name])
            local_steps += 1
    denom = max(1, local_steps) * learning_rate
    new_c_local = {}
    for name, p in model.named_parameters():
        new_c_local[name] = (c_local[name].to(device) - c_global[name].to(device)
                             + (init_params[name] - p.data) / denom).detach().cpu()
    return model, new_c_local


def feddyn_client_update(model, global_model, optimizer, data_loader, criterion,
                         device, local_epochs, alpha=0.01, prev_grads=None, amp=False):
    """FedDyn dynamic regularisation (Acar et al., 2021)."""
    model.train()
    if prev_grads is None:
        prev_grads = {name: torch.zeros_like(p.data) for name, p in model.named_parameters()}
    g_params = {name: p.data.clone() for name, p in global_model.named_parameters()}
    pg = {name: t.to(device) for name, t in prev_grads.items()}
    for _ in range(local_epochs):
        for _, (points, target) in enumerate(data_loader):
            if points.size(0) <= 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            points = points.to(device, non_blocking=True)  # augmented in the loader worker
            target = target.to(device).long()
            with autocast_ctx(device, amp):
                pred, trans_feat = model(points)
                loss = criterion(pred, target, trans_feat)
            reg = 0.0
            for name, p in model.named_parameters():
                reg = reg + torch.sum(pg[name] * p)
                reg = reg + (alpha / 2.0) * torch.sum((p - g_params[name]) ** 2)
            (loss + reg).backward()
            optimizer.step()
    grads = {}
    for name, p in model.named_parameters():
        # ``prev_grads`` may live on the model's device (it is created with
        # ``torch.zeros_like(p.data)`` by ``initialize_scaffold_control_variates``)
        # on the first round, while the new terms below are moved to CPU.  Keep the
        # whole accumulator on CPU so it (a) matches the cross-round storage contract
        # used by the FedDyn server step (which does ``.to(p.device)`` on use) and
        # (b) never mixes CUDA and CPU tensors.
        grads[name] = (prev_grads[name].detach().cpu()
                       + alpha * (p.data.detach().cpu() - g_params[name].detach().cpu()))
    return model, grads


def _extract_representation(model, points):
    """Return (logits, representation) where representation is a flat (B, D)
    feature vector taken from the final auxiliary tensor returned by the
    backbone (l3_points for PointNet++; falls back to logits otherwise)."""
    pred, aux = model(points)
    if aux is not None and aux.dim() >= 2 and aux.size(-1) == 1:
        rep = aux.view(aux.size(0), -1)
    elif aux is not None and aux.dim() == 3:
        rep = aux.mean(dim=-1).view(aux.size(0), -1)
    else:
        rep = pred
    return pred, rep


def moon_client_update(model, global_model, prev_local_model, optimizer, data_loader,
                       criterion, device, local_epochs, mu=1.0, temperature=0.5, amp=False):
    """MOON: model-contrastive federated learning (Li, He, Song -- CVPR 2021)."""
    model.train()
    global_model.eval()
    if prev_local_model is not None:
        prev_local_model.eval()
    cos = torch.nn.CosineSimilarity(dim=-1)
    for _ in range(local_epochs):
        for _, (points, target) in enumerate(data_loader):
            if points.size(0) <= 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            points = points.to(device, non_blocking=True)  # augmented in the loader worker
            target = target.to(device).long()
            with autocast_ctx(device, amp):
                pred, z = _extract_representation(model, points)
                loss = criterion(pred, target, None)
                with torch.no_grad():
                    _, z_g = _extract_representation(global_model, points)
                    z_p = z_g if prev_local_model is None else _extract_representation(prev_local_model, points)[1]
                logits = torch.stack([cos(z, z_g), cos(z, z_p)], dim=1) / temperature
                labels = torch.zeros(points.size(0), dtype=torch.long, device=device)
                contrastive = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss + mu * contrastive
            loss.backward()
            optimizer.step()
    return model


def ditto_client_update(model, global_model, personal_model, optimizer, personal_optimizer,
                        data_loader, criterion, device, local_epochs, lam=0.1, amp=False):
    """Ditto (Li, Hu, Beirami, Smith -- ICML 2021): personalised FL.

    Stage 1 trains ``model`` (returned to the server) with FedAvg-style local
    SGD.  Stage 2 trains the per-client ``personal_model`` with a proximal pull
    toward the just-updated ``model`` (strength ``lam``).
    """
    _ = global_model
    model.train()
    for _ in range(local_epochs):
        for _, (points, target) in enumerate(data_loader):
            if points.size(0) <= 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            points = points.to(device, non_blocking=True)  # augmented in the loader worker
            target = target.to(device).long()
            with autocast_ctx(device, amp):
                pred, trans_feat = model(points)
                loss = criterion(pred, target, trans_feat)
            loss.backward()
            optimizer.step()
    snapshot = {name: p.data.clone() for name, p in model.named_parameters()}
    personal_model.train()
    for _ in range(local_epochs):
        for _, (points, target) in enumerate(data_loader):
            if points.size(0) <= 1:
                continue
            personal_optimizer.zero_grad(set_to_none=True)
            points = points.to(device, non_blocking=True)  # augmented in the loader worker
            target = target.to(device).long()
            with autocast_ctx(device, amp):
                pred, trans_feat = personal_model(points)
                loss = criterion(pred, target, trans_feat)
            prox = 0.0
            for name, p in personal_model.named_parameters():
                prox = prox + torch.sum((p - snapshot[name]) ** 2)
            loss = loss + (lam / 2.0) * prox
            loss.backward()
            personal_optimizer.step()
    return model, personal_model
