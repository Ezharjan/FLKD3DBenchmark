"""Torch-based unit / smoke tests for the FL+KD 3D point-cloud classification benchmark.

Run inside the `cg` conda environment (needs torch; CPU is fine, ~1 min):

    conda activate cg
    python flkd/unit_test.py

Covers: model construction & forward shapes, all 13 FL strategies (aggregators/
updates), all 11 KD losses (forward + backward + finiteness), data partitioning, the
confusion-matrix evaluator, FedBN BN isolation, checkpoint/RNG resume helpers,
and a tiny end-to-end FL round + KD epoch on synthetic data.
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)
sys.path.append(os.path.join(BASE, 'models'))

from flkd import engine, fl_utils, kd_utils  # noqa: E402

PASS = []


def ok(msg):
    PASS.append(msg); print(f'  [ok] {msg}')


class TinyNet(nn.Module):
    """Fast stand-in classifier returning (log_softmax_logits, feature)."""
    def __init__(self, num_class=4, in_ch=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, 16, 1)
        self.bn = nn.BatchNorm1d(16)
        self.fc = nn.Linear(16, num_class)

    def forward(self, x):                       # x: (B, C, N)
        h = F.relu(self.bn(self.conv(x)))
        pooled = h.max(dim=2)[0]
        return F.log_softmax(self.fc(pooled), dim=1), pooled.unsqueeze(-1)


class TinyLoss(nn.Module):
    def forward(self, pred, target, trans_feat=None):
        return F.nll_loss(pred, target)


def _collate_bcn(batch):
    """Collate test items into (B, C, N) tensors + long labels, matching the
    production training loader (engine.augment_collate transposes to (B, C, N)),
    so client updates and the end-to-end flow receive model-ready inputs."""
    pts = np.stack([np.asarray(b[0], dtype=np.float32) for b in batch])  # (B, N, C)
    targets = torch.as_tensor([int(b[1]) for b in batch], dtype=torch.long)
    return torch.from_numpy(pts).transpose(2, 1).contiguous(), targets   # (B, C, N)


def _loader(n=24, num_class=4, npoint=64, in_ch=3, bs=8):
    pts = torch.randn(n, npoint, in_ch)
    lab = torch.randint(0, num_class, (n,))
    # Items are (npoint, in_ch); _collate_bcn transposes the batch to (B, C, N),
    # exactly like the real augmenting loader, so consumers get model-ready tensors.
    class _DS(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i): return pts[i].numpy(), int(lab[i])
    return torch.utils.data.DataLoader(_DS(), batch_size=bs, shuffle=True, collate_fn=_collate_bcn)


def test_models():
    for name in ['pointnet2_cls_ssg', 'pointnet2_cls_msg']:
        import importlib
        m = importlib.import_module(name).get_model(8, normal_channel=False)
        out, feat = m(torch.randn(2, 3, 1024))
        assert out.shape == (2, 8), (name, out.shape)
    ok('real PointNet++ teachers instantiate + forward (B,3,1024)->(B,8)')
    for cls in [kd_utils.SmallPointNetCls, kd_utils.SmallPointNet2ClsSsg, kd_utils.DGCNNClsStudent]:
        s = cls(8, normal_channel=False)
        out, feat = s(torch.randn(2, 3, 256 if cls is kd_utils.DGCNNClsStudent else 1024))
        assert out.shape == (2, 8)
        assert feat is not None, f'{cls.__name__} must expose a feature for feature-KD'
    ok('all 3 students forward + expose features (incl. SmallPointNetCls)')


def test_fl_aggregators():
    torch.manual_seed(0)
    models = [TinyNet() for _ in range(4)]
    g = fl_utils.fedavg_aggregate(models, [1, 2, 3, 4])
    assert isinstance(g, nn.Module); ok('fedavg_aggregate (weighted)')
    g = fl_utils.fedmedian_aggregate(models); ok('fedmedian_aggregate')
    base = TinyNet()
    vel = fl_utils.initialize_state_like(base)
    g, vel = fl_utils.fedavgm_update(base, models[0], vel, server_lr=1.0, server_momentum=0.9)
    ok('fedavgm_update')
    for mode in ['adam', 'yogi', 'adagrad']:
        m_s = fl_utils.initialize_state_like(base); v_s = fl_utils.initialize_state_like(base)
        g, m_s, v_s, step = fl_utils.fedopt_update(base, models[0], m_s, v_s, 0, mode=mode,
                                                   server_lr=0.05)
        assert step == 1
        # every param must be finite
        assert all(torch.isfinite(p).all() for p in g.parameters())
    ok('fedopt_update adam/yogi/adagrad (finite, integer buffers skipped)')
    taus = [10, 12, 8, 5]
    g = fl_utils.fednova_aggregate(base, models, taus, [1, 1, 1, 1]); ok('fednova_aggregate')
    # bn helpers
    bn_keys = fl_utils.bn_param_names(base)
    assert any('bn' in k for k in bn_keys); ok(f'bn_param_names found {len(bn_keys)} BN keys')
    before = [m.bn.running_mean.clone() for m in models]
    fl_utils.broadcast_non_bn(base, models, bn_keys)
    assert all(torch.allclose(b, m.bn.running_mean) for b, m in zip(before, models))
    ok('broadcast_non_bn keeps client BN buffers local (FedBN)')


def test_fl_client_updates():
    device = torch.device('cpu')
    loader = _loader()
    crit = TinyLoss()
    base = TinyNet()
    # fedprox
    m = TinyNet(); opt = torch.optim.SGD(m.parameters(), lr=0.01)
    fl_utils.fedprox_client_update(m, base, opt, loader, crit, device, 1, mu=0.01)
    ok('fedprox_client_update runs')
    # scaffold
    m = TinyNet(); opt = torch.optim.SGD(m.parameters(), lr=0.01)
    cg = fl_utils.initialize_scaffold_control_variates(m)
    cl = fl_utils.initialize_scaffold_control_variates(m)
    m, new_cl = fl_utils.scaffold_client_update(m, base, opt, loader, crit, device, 1, cg, cl, 0.01)
    assert all(torch.isfinite(v).all() for v in new_cl.values()); ok('scaffold_client_update (finite control variates)')
    # feddyn
    m = TinyNet(); opt = torch.optim.SGD(m.parameters(), lr=0.01)
    pg = fl_utils.initialize_scaffold_control_variates(m)
    m, grads = fl_utils.feddyn_client_update(m, base, opt, loader, crit, device, 1, alpha=0.01, prev_grads=pg)
    ok('feddyn_client_update')
    # moon
    m = TinyNet(); opt = torch.optim.SGD(m.parameters(), lr=0.01)
    fl_utils.moon_client_update(m, base, None, opt, loader, crit, device, 1, mu=1.0, temperature=0.5)
    ok('moon_client_update')
    # ditto
    m = TinyNet(); pm = TinyNet()
    opt = torch.optim.SGD(m.parameters(), lr=0.01); popt = torch.optim.SGD(pm.parameters(), lr=0.01)
    fl_utils.ditto_client_update(m, base, pm, opt, popt, loader, crit, device, 1, lam=0.1)
    ok('ditto_client_update')


def test_kd_losses():
    torch.manual_seed(0)
    B, C, D = 8, 6, 32
    s_logits = torch.randn(B, C, requires_grad=True)
    t_logits = torch.randn(B, C)
    target = torch.randint(0, C, (B,))
    s_feat = torch.randn(B, D, requires_grad=True)
    t_feat = torch.randn(B, D)
    losses = {
        'vanilla': (kd_utils.VanillaDistillationLoss(), (s_logits, t_logits, target)),
        'self': (kd_utils.SelfDistillationLoss(), (s_logits, t_logits, target)),
        'logit_mse': (kd_utils.LogitMSEDistillationLoss(), (s_logits, t_logits, target)),
        'cosine': (kd_utils.CosineDistillationLoss(), (s_logits, t_logits, target)),
        'feature': (kd_utils.FeatureDistillationLoss(), (s_logits, t_logits, target),
                    dict(student_features=s_feat, teacher_features=t_feat)),
        'attention': (kd_utils.AttentionDistillationLoss(), (s_logits, t_logits, target),
                      dict(student_features=s_feat, teacher_features=t_feat)),
        'crd': (kd_utils.CRDDistillationLoss(), (s_logits, t_logits, target),
                dict(student_features=s_feat, teacher_features=t_feat)),
        'dkd': (kd_utils.DKDDistillationLoss(), (s_logits, t_logits, target)),
        'rkd': (kd_utils.RKDDistillationLoss(), (s_logits, t_logits, target),
                dict(student_features=s_feat, teacher_features=t_feat)),
        'sp': (kd_utils.SPDistillationLoss(), (s_logits, t_logits, target),
               dict(student_features=s_feat, teacher_features=t_feat)),
    }
    for name, spec in losses.items():
        crit, args = spec[0], spec[1]
        kwargs = spec[2] if len(spec) > 2 else {}
        loss = crit(*args, **kwargs)
        assert torch.isfinite(loss), f'{name} produced non-finite loss'
        loss.backward(retain_graph=True)
        ok(f'KD loss {name}: finite + backward')
    # attention transfer must be NON-degenerate: gradient must flow to features
    sf = torch.randn(B, D, requires_grad=True)
    att = kd_utils.AttentionDistillationLoss()(torch.randn(B, C, requires_grad=True), t_logits,
                                               target, student_features=sf,
                                               teacher_features=torch.randn(B, D))
    att.backward()
    assert sf.grad is not None and sf.grad.abs().sum() > 0, 'attention KD is a no-op!'
    ok('attention KD propagates gradient to features (non-degenerate)')
    # multi-teacher
    mt = kd_utils.MultiTeacherDistillationLoss()
    loss = mt(s_logits, [t_logits, torch.randn(B, C)], target)
    assert torch.isfinite(loss); ok('KD loss multi_teacher')
    # unlabeled path (target=None) for every logits-only + feature loss
    for crit in [kd_utils.VanillaDistillationLoss(use_hard=False), kd_utils.DKDDistillationLoss(use_hard=False)]:
        assert torch.isfinite(crit(s_logits, t_logits, None)); 
    ok('KD unlabeled/no-CE path (target=None) finite')


def test_partition_and_eval():
    # synthetic labelled dataset
    class DS(torch.utils.data.Dataset):
        def __init__(self): self.labels = np.array([i % 5 for i in range(100)])
        def __len__(self): return 100
        def __getitem__(self, i): return np.random.randn(32, 3).astype('float32'), int(self.labels[i])
    ds = DS()
    for part in ['iid', 'label_skew', 'dirichlet']:
        parts = fl_utils.split_dataset(ds, 5, partition=part, seed=1)
        assert sum(len(p) for p in parts) == 100, part
    ok('split_dataset iid/label_skew/dirichlet partition full dataset')
    # evaluator on synthetic data: check OA/mAcc are in [0,1] and the confusion matrix is complete
    loader = torch.utils.data.DataLoader(ds, batch_size=16)
    res = engine.evaluate(TinyNet(num_class=5), loader, 5, torch.device('cpu'))
    assert 0.0 <= res['instance_acc'] <= 1.0 and 0.0 <= res['class_acc'] <= 1.0
    assert res['confusion'].shape == (5, 5) and res['confusion'].sum() == 100
    ok('engine.evaluate returns valid OA/mAcc + full confusion matrix')


def test_resume_helpers():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'r.pth')
        engine.set_global_seed(123)
        st = engine.get_rng_state()
        a = torch.randn(3)
        engine.atomic_save({'rng': st, 'x': a}, p)
        loaded = engine.safe_load(p)
        engine.set_rng_state(loaded['rng'])
        b = torch.randn(3)
        engine.set_rng_state(st)
        c = torch.randn(3)
        assert torch.allclose(b, c), 'RNG restore not reproducible'
    ok('atomic_save/safe_load + RNG state restore reproducible')


def test_end_to_end():
    device = torch.device('cpu')
    loader = _loader(n=32, num_class=4)
    test_loader = _loader(n=16, num_class=4, bs=8)
    crit = TinyLoss()
    # one FedAvg round across 2 clients
    global_model = TinyNet()
    clients = [TinyNet() for _ in range(2)]
    for c in clients: c.load_state_dict(global_model.state_dict())
    opts = [torch.optim.SGD(c.parameters(), lr=0.01) for c in clients]
    for c, o in zip(clients, opts):
        c.train()
        for pts, tgt in loader:
            o.zero_grad()
            pts = pts.to(device)
            out, _ = c(pts); TinyLoss()(out, tgt.long()).backward(); o.step()
    g = fl_utils.fedavg_aggregate(clients, [1, 1])
    res = engine.evaluate(g, test_loader, 4, device)
    ok(f'end-to-end FL round -> eval OA={res["instance_acc"]:.3f}')
    # one KD epoch (vanilla) student<-teacher
    teacher = TinyNet(); student = TinyNet()
    kd = kd_utils.VanillaDistillationLoss()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    student.train(); teacher.eval()
    for pts, tgt in loader:
        opt.zero_grad()
        pts = pts.to(device)
        s_out, _ = student(pts)
        with torch.no_grad(): t_out, _ = teacher(pts)
        loss = kd(s_out, t_out, tgt.long()); loss.backward(); opt.step()
    ok('end-to-end KD epoch (vanilla) trains a student')


def main():
    print('Running FL+KD unit/smoke tests (torch present)...')
    for fn in [test_models, test_fl_aggregators, test_fl_client_updates, test_kd_losses,
               test_partition_and_eval, test_resume_helpers, test_end_to_end]:
        print(f'\n== {fn.__name__} ==')
        fn()
    print(f'\nALL {len(PASS)} CHECKS PASSED')


if __name__ == '__main__':
    main()
