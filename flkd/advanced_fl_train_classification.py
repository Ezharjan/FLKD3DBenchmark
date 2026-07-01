"""Advanced Federated Learning for PointNet/PointNet++ Classification.

Implements 13 federated-learning strategies (the CLI also accepts ``vanilla``
as an alias of FedAvg) with bf16 AMP, multi-worker data loading, confusion-matrix
evaluation, dataset-namespaced outputs and per-round auto-resume.

Strategies: fedavg/vanilla, fedprox, scaffold, feddyn, fedavgm, fedadam,
fedyogi, fedadagrad, fedmedian, fedbn, moon, ditto, fednova.
"""

import argparse
import copy
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

from data_utils.dataset_factory import create_classification_datasets  # noqa: E402
from flkd import engine  # noqa: E402
from flkd.fl_utils import (  # noqa: E402
    split_dataset, fedavg_aggregate, fedprox_client_update, scaffold_client_update,
    feddyn_client_update, initialize_scaffold_control_variates,
    aggregate_scaffold_control_variates, fedmedian_aggregate, initialize_state_like,
    fedavgm_update, fedopt_update, moon_client_update, ditto_client_update,
    fednova_aggregate, count_local_steps, state_dict_size_mb, per_round_communication_mb,
    bn_param_names, broadcast_non_bn,
)
from flkd.kd_utils import get_model_size, measure_inference_time  # noqa: E402
from flkd.engine import augment_points, autocast_ctx  # noqa: E402
from experiment import prepare_experiment  # noqa: E402

FL_STRATEGIES = ['fedavg', 'vanilla', 'fedprox', 'scaffold', 'feddyn',
                 'fedavgm', 'fedadam', 'fedyogi', 'fedadagrad', 'fedmedian',
                 'fedbn', 'moon', 'ditto', 'fednova']
ADAPTIVE = {'fedadam': 'adam', 'fedyogi': 'yogi', 'fedadagrad': 'adagrad'}


def parse_args():
    p = argparse.ArgumentParser('Federated Learning for PointNet Classification')
    p.add_argument('--use_cpu', action='store_true', default=False)
    p.add_argument('--no_amp', action='store_true', default=False, help='disable bf16 autocast')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--num_workers', type=int, default=-1, help='-1 = auto from CPU allocation')
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--model', default='pointnet2_cls_ssg')
    p.add_argument('--num_category', default=None, type=int)
    p.add_argument('--dataset', type=str, default='modelnet40')
    p.add_argument('--data_root', type=str, default='data')
    p.add_argument('--output_root', type=str, default='outputs',
                   help='results go to <output_root>/<dataset>/flkd/federated/classification/<log_dir>')
    p.add_argument('--train_split', type=float, default=0.8)
    p.add_argument('--dataset_seed', type=int, default=0)
    p.add_argument('--cache_dataset', dest='cache_dataset', action='store_true', default=True)
    p.add_argument('--no_cache', dest='cache_dataset', action='store_false')
    p.add_argument('--min_class_samples', type=int, default=1)
    p.add_argument('--epoch', default=200, type=int)
    p.add_argument('--learning_rate', default=0.001, type=float)
    p.add_argument('--num_point', type=int, default=1024)
    p.add_argument('--optimizer', type=str, default='Adam')
    p.add_argument('--log_dir', type=str, default=None)
    p.add_argument('--decay_rate', type=float, default=1e-4)
    p.add_argument('--use_normals', action='store_true', default=False)
    p.add_argument('--process_data', action='store_true', default=False)
    p.add_argument('--use_uniform_sample', action='store_true', default=False)
    p.add_argument('--no_resume', dest='resume', action='store_false', default=True,
                   help='ignore any existing resume.pth and start fresh')

    p.add_argument('--fl_strategy', type=str, default='fedavg', choices=FL_STRATEGIES)
    p.add_argument('--num_clients', type=int, default=5)
    p.add_argument('--local_epochs', type=int, default=5)
    p.add_argument('--communication_rounds', type=int, default=20)
    p.add_argument('--iid', action='store_true', default=False)
    p.add_argument('--partition', type=str, default='label_skew',
                   choices=['iid', 'label_skew', 'dirichlet'])
    p.add_argument('--dirichlet_alpha', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--mu', type=float, default=0.01, help='proximal term for FedProx')
    p.add_argument('--alpha', type=float, default=0.01, help='regularisation for FedDyn')
    p.add_argument('--server_lr', type=float, default=1.0, help='server LR for FedAvgM')
    p.add_argument('--adaptive_server_lr', type=float, default=0.05,
                   help='server LR for FedAdam/FedYogi/FedAdagrad')
    p.add_argument('--server_momentum', type=float, default=0.9)
    p.add_argument('--server_beta1', type=float, default=0.9)
    p.add_argument('--server_beta2', type=float, default=0.999)
    p.add_argument('--server_eps', type=float, default=1e-3)
    p.add_argument('--moon_temperature', type=float, default=0.5)
    p.add_argument('--moon_mu', type=float, default=1.0)
    p.add_argument('--ditto_lambda', type=float, default=0.1)
    p.add_argument('--max_runtime_hours', type=float, default=0.0,
                   help='graceful wall-clock budget in hours (0 = unlimited); the trainer finishes '
                        'the current communication round, checkpoints, and exits cleanly before this '
                        'budget so a time-limited job resumes with no lost work')
    return p.parse_args()


def make_optimizer(args, model):
    if args.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)


def train_client_fedavg(model, optimizer, data_loader, criterion, device, local_epochs, amp):
    """Plain local SGD pass (FedAvg / FedAvgM / FedAdam family / FedMedian / FedNova / FedBN)."""
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
    return model


def _build_models(args, model_module, num_class, device):
    global_model = model_module.get_model(num_class, normal_channel=args.use_normals).to(device)
    client_models = []
    for _ in range(args.num_clients):
        m = model_module.get_model(num_class, normal_channel=args.use_normals).to(device)
        m.load_state_dict(global_model.state_dict())
        client_models.append(m)
    return global_model, client_models


def main(args):
    engine.set_global_seed(args.seed)
    engine.setup_perf(use_cpu=args.use_cpu)
    amp = engine.amp_default(args)

    exp = prepare_experiment(
        subdirs=[args.dataset, 'flkd', 'federated', 'classification'],
        log_dir=args.log_dir,
        name_hint=f"fl_{args.model}_{args.fl_strategy}",
        output_root=Path(args.output_root),
        logger_name_prefix='fl',
    )
    logger = exp.logger

    def log_string(msg):
        logger.info(msg)
        print(msg)

    # FedNova's tau-normalisation is only valid for plain SGD locals; switch the
    # optimiser BEFORE config.json is written so the logged optimiser is accurate.
    if args.fl_strategy == 'fednova' and args.optimizer == 'Adam':
        args.optimizer = 'SGD'
    exp.log_args(args)
    device = engine.pick_device(args.use_cpu)
    if args.fl_strategy == 'fednova':
        log_string('FedNova: using SGD client optimiser (tau-normalisation requires SGD locals).')
    log_string(f'Using device: {device}; AMP(bf16)={amp}; strategy={args.fl_strategy}')

    log_string('Load dataset ...')
    train_dataset, test_dataset, num_class, _ = create_classification_datasets(args)
    num_class = args.num_category or num_class
    num_workers = engine.resolve_num_workers(args)

    test_loader = engine.build_loader(test_dataset, args.batch_size, False, num_workers)

    model_module = importlib.import_module(args.model)
    global_model, client_models = _build_models(args, model_module, num_class, device)
    criterion = model_module.get_loss().to(device)
    client_optimizers = [make_optimizer(args, m) for m in client_models]

    log_string(f'Splitting {len(train_dataset)} samples across {args.num_clients} clients '
               f'(iid={args.iid}, partition={args.partition}, alpha={args.dirichlet_alpha})')
    client_datasets = split_dataset(train_dataset, args.num_clients, iid=args.iid,
                                    partition=args.partition, dirichlet_alpha=args.dirichlet_alpha,
                                    seed=args.seed)
    client_loaders = [engine.build_loader(ds, args.batch_size, True, num_workers,
                                          drop_last=False, augment=True)
                      for ds in client_datasets]
    client_weights = [len(ds) for ds in client_datasets]

    # ---- strategy-specific state ----
    strat = args.fl_strategy
    c_global = c_locals = client_grads = prev_local_models = None
    personal_models = personal_optimizers = None
    bn_keys = bn_param_names(global_model) if strat == 'fedbn' else set()
    if strat == 'scaffold':
        c_global = initialize_scaffold_control_variates(global_model)
        c_locals = [initialize_scaffold_control_variates(m) for m in client_models]
    elif strat == 'feddyn':
        client_grads = [initialize_scaffold_control_variates(m) for m in client_models]
    elif strat == 'moon':
        prev_local_models = [None] * args.num_clients
    elif strat == 'ditto':
        personal_models, personal_optimizers = [], []
        for _ in range(args.num_clients):
            pm = model_module.get_model(num_class, normal_channel=args.use_normals).to(device)
            pm.load_state_dict(global_model.state_dict())
            personal_models.append(pm)
            personal_optimizers.append(make_optimizer(args, pm))

    velocity_state = initialize_state_like(global_model, 0.0)
    m_state = initialize_state_like(global_model, 0.0)
    v_state = initialize_state_like(global_model, 0.0)
    server_step = 0
    best_instance_acc = best_class_acc = 0.0
    best_epoch = 0
    instance_acc = class_acc = 0.0
    start_round = 0

    payload_mb = state_dict_size_mb(global_model)
    comm_mb_round = per_round_communication_mb(global_model, args.num_clients, bidirectional=True)
    log_string(f'Model payload: {payload_mb:.2f} MB; comm/round ~{comm_mb_round:.2f} MB.')

    # ---- auto-resume ----
    resume_path = exp.checkpoints_dir / 'resume.pth'
    if args.resume and resume_path.exists():
        try:
            st = engine.safe_load(resume_path, map_location=device)
            start_round = st['round']
            global_model.load_state_dict(st['global_model'])
            for m, sd in zip(client_models, st['client_models']):
                m.load_state_dict(sd)
            for o, sd in zip(client_optimizers, st['client_optimizers']):
                o.load_state_dict(sd)
            best_instance_acc = st['best_instance_acc']; best_class_acc = st['best_class_acc']
            best_epoch = st['best_epoch']
            velocity_state = st.get('velocity_state', velocity_state)
            m_state = st.get('m_state', m_state); v_state = st.get('v_state', v_state)
            server_step = st.get('server_step', 0)
            if strat == 'scaffold':
                c_global = st['c_global']; c_locals = st['c_locals']
            elif strat == 'feddyn':
                client_grads = st['client_grads']
            elif strat == 'moon':
                prev_local_models = []
                for sd in st['prev_local_models']:
                    if sd is None:
                        prev_local_models.append(None)
                    else:
                        pm = model_module.get_model(num_class, normal_channel=args.use_normals).to(device)
                        pm.load_state_dict(sd); pm.eval()
                        prev_local_models.append(pm)
            elif strat == 'ditto':
                for pm, sd in zip(personal_models, st['personal_models']):
                    pm.load_state_dict(sd)
                for po, sd in zip(personal_optimizers, st['personal_optimizers']):
                    po.load_state_dict(sd)
            engine.set_rng_state(st.get('rng'))
            log_string(f'>> Resumed from round {start_round}/{args.communication_rounds} '
                       f'(best so far {best_instance_acc:.4f}).')
        except Exception as exc:
            log_string(f'>> Resume failed ({exc}); starting fresh.')
            start_round = 0

    if start_round >= args.communication_rounds:
        log_string('Training already complete; nothing to do.')
        if not (exp.checkpoints_dir / 'last_model.pth').exists():
            engine.atomic_save({
                'epoch': args.communication_rounds, 'instance_acc': best_instance_acc,
                'class_acc': best_class_acc, 'model_state_dict': global_model.state_dict(),
                'strategy': strat, 'seed': args.seed, 'dataset': args.dataset,
            }, exp.checkpoints_dir / 'last_model.pth')
        return

    log_string('Start federated training...')
    run_start = time.time()
    budget = args.max_runtime_hours * 3600.0 if args.max_runtime_hours and args.max_runtime_hours > 0 else None
    last_round_seconds = 0.0
    for round_idx in range(start_round, args.communication_rounds):
        if budget is not None and (time.time() - run_start) + 1.2 * last_round_seconds > budget:
            log_string('>> Wall-clock budget (%.2fh) reached; stopping cleanly at round %d/%d. '
                       'Re-run to resume.' % (args.max_runtime_hours, round_idx, args.communication_rounds))
            return
        log_string('Communication round %d/%d' % (round_idx + 1, args.communication_rounds))
        round_start = time.time()
        client_taus = []

        for cid in range(args.num_clients):
            if strat in {'fedavg', 'vanilla', 'fedavgm', 'fedadam', 'fedyogi',
                         'fedadagrad', 'fedmedian', 'fednova', 'fedbn'}:
                train_client_fedavg(client_models[cid], client_optimizers[cid],
                                    client_loaders[cid], criterion, device, args.local_epochs, amp)
            elif strat == 'fedprox':
                fedprox_client_update(client_models[cid], global_model, client_optimizers[cid],
                                     client_loaders[cid], criterion, device, args.local_epochs,
                                     mu=args.mu, amp=amp)
            elif strat == 'scaffold':
                client_models[cid], c_locals[cid] = scaffold_client_update(
                    client_models[cid], global_model, client_optimizers[cid], client_loaders[cid],
                    criterion, device, args.local_epochs, c_global, c_locals[cid],
                    args.learning_rate, amp=amp)
            elif strat == 'feddyn':
                client_models[cid], client_grads[cid] = feddyn_client_update(
                    client_models[cid], global_model, client_optimizers[cid], client_loaders[cid],
                    criterion, device, args.local_epochs, alpha=args.alpha,
                    prev_grads=client_grads[cid], amp=amp)
            elif strat == 'moon':
                moon_client_update(client_models[cid], global_model, prev_local_models[cid],
                                  client_optimizers[cid], client_loaders[cid], criterion, device,
                                  args.local_epochs, mu=args.moon_mu,
                                  temperature=args.moon_temperature, amp=amp)
                prev_local_models[cid] = copy.deepcopy(client_models[cid]).eval()
            elif strat == 'ditto':
                client_models[cid], personal_models[cid] = ditto_client_update(
                    client_models[cid], global_model, personal_models[cid], client_optimizers[cid],
                    personal_optimizers[cid], client_loaders[cid], criterion, device,
                    args.local_epochs, lam=args.ditto_lambda, amp=amp)
            if strat == 'fednova':
                client_taus.append(count_local_steps(client_loaders[cid], args.local_epochs))

        # ---- aggregation ----
        if strat in {'fedavg', 'vanilla', 'fedprox', 'moon', 'ditto'}:
            # Ditto: the server-side global model is FedAvg of the global copies;
            # the per-client personalised models are kept but evaluated separately
            # is out of scope for the shared-test-set protocol (see README note).
            global_model = fedavg_aggregate(client_models, client_weights)
        elif strat == 'feddyn':
            # FedDyn server step (Acar et al., 2021): average the client models and
            # add the dynamic correction (1/alpha) * mean_i(h_i), where the h_i are
            # the per-client accumulated gradient states returned by the client
            # update.
            global_model = fedavg_aggregate(client_models, client_weights)
            n_cl = len(client_grads)
            sd = global_model.state_dict()
            for name, p in global_model.named_parameters():
                h_mean = sum(cg[name].float().to(p.device) for cg in client_grads) / n_cl
                sd[name] = (sd[name].float() + (1.0 / args.alpha) * h_mean).to(sd[name].dtype)
            global_model.load_state_dict(sd)
        elif strat == 'scaffold':
            global_model = fedavg_aggregate(client_models, client_weights)
            c_global = aggregate_scaffold_control_variates(c_locals, client_weights)
        elif strat == 'fedmedian':
            global_model = fedmedian_aggregate(client_models)
        elif strat == 'fedbn':
            # NOTE: FedBN keeps BN client-local; this global model (BN averaged) is a
            # shared-test-set proxy for reporting (see README 'FedBN evaluation note').
            global_model = fedavg_aggregate(client_models, client_weights)
        elif strat == 'fedavgm':
            agg = fedavg_aggregate(client_models, client_weights)
            global_model, velocity_state = fedavgm_update(
                global_model, agg, velocity_state,
                server_lr=args.server_lr, server_momentum=args.server_momentum)
        elif strat in ADAPTIVE:
            agg = fedavg_aggregate(client_models, client_weights)
            global_model, m_state, v_state, server_step = fedopt_update(
                global_model, agg, m_state, v_state, server_step, mode=ADAPTIVE[strat],
                server_lr=args.adaptive_server_lr, beta1=args.server_beta1,
                beta2=args.server_beta2, eps=args.server_eps)
        elif strat == 'fednova':
            global_model = fednova_aggregate(global_model, client_models, client_taus, client_weights)

        # ---- broadcast to clients ----
        if strat == 'fedbn':
            broadcast_non_bn(global_model, client_models, bn_keys)  # keep client-local BN
        else:
            for cid in range(args.num_clients):
                client_models[cid].load_state_dict(global_model.state_dict())

        # ---- evaluate ----
        res = engine.evaluate(global_model, test_loader, num_class, device, amp=False)  # fp32 evaluation
        instance_acc, class_acc = res['instance_acc'], res['class_acc']
        round_time = time.time() - round_start
        log_string('Test Instance Acc: %.4f, Class Acc: %.4f, Round time: %.1fs'
                   % (instance_acc, class_acc, round_time))
        exp.record_metrics(split='eval', epoch=round_idx + 1,
                           instance_acc=instance_acc, class_acc=class_acc,
                           best_instance_acc=best_instance_acc, best_class_acc=best_class_acc,
                           round_seconds=round_time, comm_mb=comm_mb_round)

        if instance_acc > best_instance_acc:
            best_instance_acc, best_class_acc, best_epoch = instance_acc, class_acc, round_idx + 1
            engine.atomic_save({
                'epoch': best_epoch, 'instance_acc': instance_acc, 'class_acc': class_acc,
                'model_state_dict': global_model.state_dict(), 'strategy': strat,
                'seed': args.seed, 'partition': args.partition,
                'dirichlet_alpha': args.dirichlet_alpha, 'dataset': args.dataset,
            }, exp.checkpoints_dir / 'best_model.pth')
        log_string('Best Instance %.4f / Class %.4f (round %d)'
                   % (best_instance_acc, best_class_acc, best_epoch))

        # ---- write resume checkpoint (atomic) ----
        resume_state = {
            'round': round_idx + 1,
            'global_model': global_model.state_dict(),
            'client_models': [m.state_dict() for m in client_models],
            'client_optimizers': [o.state_dict() for o in client_optimizers],
            'best_instance_acc': best_instance_acc, 'best_class_acc': best_class_acc,
            'best_epoch': best_epoch, 'velocity_state': velocity_state,
            'm_state': m_state, 'v_state': v_state, 'server_step': server_step,
            'rng': engine.get_rng_state(),
        }
        if strat == 'scaffold':
            resume_state['c_global'] = c_global; resume_state['c_locals'] = c_locals
        elif strat == 'feddyn':
            resume_state['client_grads'] = client_grads
        elif strat == 'moon':
            resume_state['prev_local_models'] = [None if m is None else m.state_dict()
                                                 for m in prev_local_models]
        elif strat == 'ditto':
            resume_state['personal_models'] = [m.state_dict() for m in personal_models]
            resume_state['personal_optimizers'] = [o.state_dict() for o in personal_optimizers]
        engine.atomic_save(resume_state, resume_path)
        last_round_seconds = round_time

    # ---- finalise ----
    engine.atomic_save({
        'epoch': args.communication_rounds, 'instance_acc': instance_acc, 'class_acc': class_acc,
        'model_state_dict': global_model.state_dict(), 'strategy': strat, 'seed': args.seed,
        'dataset': args.dataset,
    }, exp.checkpoints_dir / 'last_model.pth')
    log_string('End of training. Best %.4f at round %d.' % (best_instance_acc, best_epoch))

    ckpt = engine.safe_load(exp.checkpoints_dir / 'best_model.pth', map_location=device)
    global_model.load_state_dict(ckpt['model_state_dict'])
    model_size = get_model_size(global_model)
    inference_time = measure_inference_time(global_model, test_loader, device)
    log_string('Model size: %.2f MB; Inference: %.2f ms' % (model_size, inference_time))
    exp.record_metrics(split='summary', best_instance_acc=best_instance_acc,
                       best_class_acc=best_class_acc, last_instance_acc=instance_acc,
                       last_class_acc=class_acc, model_size_mb=model_size,
                       inference_time_ms=inference_time, comm_mb_per_round=comm_mb_round,
                       num_classes=num_class)


if __name__ == '__main__':
    args = parse_args()
    if args.batch_size < 2:
        print('WARNING: Batch size raised to 2 for BatchNorm.')
        args.batch_size = 2
    main(args)
