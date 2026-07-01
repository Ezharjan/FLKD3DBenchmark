"""Advanced Knowledge Distillation for PointNet/PointNet++ Classification.

11 KD losses, 3 student backbones, bf16 AMP, multi-worker loaders,
confusion-matrix evaluation, dataset-namespaced outputs and per-epoch
auto-resume.  Supports optional ablations: --use_hard_labels off, --unlabeled,
--temperature_sweep, and heterogeneous DGCNN students.
"""

import os
import sys
import time
import copy
import argparse
import importlib
from pathlib import Path

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

from data_utils.dataset_factory import create_classification_datasets  # noqa: E402
from flkd import engine  # noqa: E402
from flkd.engine import augment_points, autocast_ctx  # noqa: E402
from flkd.kd_utils import (  # noqa: E402
    VanillaDistillationLoss, FeatureDistillationLoss, AttentionDistillationLoss,
    MultiTeacherDistillationLoss, SelfDistillationLoss, LogitMSEDistillationLoss,
    CosineDistillationLoss, CRDDistillationLoss, DKDDistillationLoss,
    RKDDistillationLoss, SPDistillationLoss,
    SmallPointNetCls, SmallPointNet2ClsSsg, DGCNNClsStudent,
    get_model_size, measure_inference_time,
)
from experiment import prepare_experiment  # noqa: E402

KD_STRATEGIES = ['vanilla', 'basic', 'feature', 'attention', 'multi_teacher', 'self',
                 'logit_mse', 'cosine', 'crd', 'dkd', 'rkd', 'sp']
FEATURE_KD = {'feature', 'attention', 'crd', 'rkd', 'sp'}
STUDENT_CHOICES = ['small_pointnet', 'small_pointnet2', 'dgcnn']


def parse_args():
    p = argparse.ArgumentParser('Knowledge Distillation for PointNet Classification')
    p.add_argument('--use_cpu', action='store_true', default=False)
    p.add_argument('--no_amp', action='store_true', default=False)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--num_workers', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--teacher_model', default='pointnet2_cls_ssg')
    p.add_argument('--student_model', default='small_pointnet2', choices=STUDENT_CHOICES)
    p.add_argument('--num_category', default=None, type=int)
    p.add_argument('--dataset', type=str, default='modelnet40')
    p.add_argument('--data_root', type=str, default='data')
    p.add_argument('--output_root', type=str, default='outputs')
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
    p.add_argument('--no_resume', dest='resume', action='store_false', default=True)

    p.add_argument('--kd_strategy', type=str, default='vanilla', choices=KD_STRATEGIES)
    p.add_argument('--alpha', type=float, default=0.5, help='soft-target weight')
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--beta', type=float, default=0.5, help='feature/attention/relational weight')
    p.add_argument('--dkd_alpha', type=float, default=1.0, help='DKD TCKD weight')
    p.add_argument('--dkd_beta', type=float, default=8.0, help='DKD NCKD weight')
    p.add_argument('--crd_temperature', type=float, default=0.07)
    p.add_argument('--teacher_path', type=str, default=None)
    p.add_argument('--teacher_paths', type=str, nargs='+', default=None)
    p.add_argument('--teacher_weights', type=float, nargs='+', default=None)

    p.add_argument('--use_hard_labels', type=str, default='on', choices=['on', 'off'])
    p.add_argument('--unlabeled', action='store_true', default=False)
    p.add_argument('--temperature_sweep', type=float, nargs='+', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_runtime_hours', type=float, default=0.0,
                   help='graceful wall-clock budget in hours (0 = unlimited); the trainer finishes '
                        'the current epoch, checkpoints, and exits cleanly before this budget so a '
                        'time-limited job resumes with no lost work')
    return p.parse_args()


def build_student(args, num_class, device):
    if args.student_model == 'small_pointnet':
        return SmallPointNetCls(num_class, normal_channel=args.use_normals).to(device)
    if args.student_model == 'dgcnn':
        return DGCNNClsStudent(num_class, normal_channel=args.use_normals).to(device)
    return SmallPointNet2ClsSsg(num_class, normal_channel=args.use_normals).to(device)


def get_distillation_criterion(args, use_hard):
    s = args.kd_strategy
    if s in {'vanilla', 'basic'}:
        return VanillaDistillationLoss(alpha=1.0 if not use_hard else args.alpha,
                                       temperature=args.temperature, use_hard=use_hard)
    if s == 'feature':
        return FeatureDistillationLoss(args.alpha, args.temperature, args.beta, use_hard)
    if s == 'attention':
        return AttentionDistillationLoss(args.alpha, args.temperature, args.beta, use_hard)
    if s == 'multi_teacher':
        return MultiTeacherDistillationLoss(args.alpha, args.temperature, args.teacher_weights, use_hard)
    if s == 'self':
        return SelfDistillationLoss(args.alpha, args.temperature, use_hard)
    if s == 'logit_mse':
        return LogitMSEDistillationLoss(args.alpha, use_hard)
    if s == 'cosine':
        return CosineDistillationLoss(args.alpha, use_hard)
    if s == 'crd':
        return CRDDistillationLoss(alpha=args.alpha, temperature=args.crd_temperature, use_hard=use_hard)
    if s == 'dkd':
        return DKDDistillationLoss(alpha=args.dkd_alpha, beta=args.dkd_beta,
                                   temperature=args.temperature, use_hard=use_hard)
    if s == 'rkd':
        return RKDDistillationLoss(args.alpha, args.beta, args.temperature, use_hard)
    if s == 'sp':
        return SPDistillationLoss(args.alpha, args.beta, args.temperature, use_hard)
    raise ValueError(f'Unknown distillation strategy: {s}')


def _forward_loss(args, criterion, student_model, teacher_model, teacher_models,
                  points, target_for_loss, device, amp):
    with autocast_ctx(device, amp):
        student_logits, student_features = student_model(points)
        if args.kd_strategy == 'multi_teacher':
            with torch.no_grad():
                teacher_logits_list = [tm(points)[0] for tm in teacher_models]
            return criterion(student_logits, teacher_logits_list, target_for_loss)
        with torch.no_grad():
            teacher_logits, teacher_features = teacher_model(points)
        if args.kd_strategy in FEATURE_KD:
            return criterion(student_logits, teacher_logits, target_for_loss,
                             student_features=student_features, teacher_features=teacher_features)
        return criterion(student_logits, teacher_logits, target_for_loss)


def train_one(args, device, amp, num_class, train_loader, test_loader,
              teacher_model, teacher_models):
    exp = prepare_experiment(
        subdirs=[args.dataset, 'flkd', 'knowledge_distillation', 'classification'],
        log_dir=args.log_dir,
        name_hint=f"kd_{args.teacher_model}_{args.student_model}_{args.kd_strategy}",
        output_root=Path(args.output_root), logger_name_prefix='kd')
    logger = exp.logger

    def log_string(m):
        logger.info(m); print(m)

    exp.log_args(args)
    use_hard = (args.use_hard_labels == 'on') and not args.unlabeled
    log_string(f'KD={args.kd_strategy} student={args.student_model} AMP={amp} '
               f'use_hard={use_hard} unlabeled={args.unlabeled} T={args.temperature}')

    student_model = build_student(args, num_class, device)
    optimizer = (torch.optim.Adam(student_model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=args.decay_rate)
                 if args.optimizer == 'Adam'
                 else torch.optim.SGD(student_model.parameters(), lr=args.learning_rate, momentum=0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    criterion = get_distillation_criterion(args, use_hard).to(device)

    # Materialise lazy KD adapters with one warm-up forward so they can be saved/
    # restored on resume and added to the optimiser deterministically.
    if args.kd_strategy in FEATURE_KD:
        warm = next(iter(train_loader), None)
        if warm is not None:
            wp = warm[0].to(device)  # train_loader already yields augmented (B, C, N)
            with torch.no_grad():
                _forward_loss(args, criterion, student_model, teacher_model, teacher_models,
                              wp, None, device, amp)
    extra = criterion.extra_parameters() if hasattr(criterion, 'extra_parameters') else []
    if extra:
        optimizer.add_param_group({'params': extra})
        log_string(f'Added {sum(p.numel() for p in extra)} KD-adapter params to optimiser.')

    best_acc = best_class_acc = 0.0
    best_epoch = 0
    start_epoch = 0
    instance_acc = class_acc = 0.0

    resume_path = exp.checkpoints_dir / 'resume.pth'
    if args.resume and resume_path.exists():
        try:
            st = engine.safe_load(resume_path, map_location=device)
            start_epoch = st['epoch']
            student_model.load_state_dict(st['student'])
            optimizer.load_state_dict(st['optimizer'])
            scheduler.load_state_dict(st['scheduler'])
            if st.get('criterion') and len(st['criterion']):
                criterion.load_state_dict(st['criterion'])
            best_acc = st['best_acc']; best_class_acc = st['best_class_acc']; best_epoch = st['best_epoch']
            engine.set_rng_state(st.get('rng'))
            log_string(f'>> Resumed from epoch {start_epoch}/{args.epoch} (best {best_acc:.4f}).')
        except Exception as exc:
            log_string(f'>> Resume failed ({exc}); starting fresh.')
            start_epoch = 0

    if start_epoch >= args.epoch:
        log_string('KD training already complete; nothing to do.')
        if not (exp.checkpoints_dir / 'last_model.pth').exists():
            engine.atomic_save({
                'epoch': args.epoch, 'instance_acc': best_acc, 'class_acc': best_class_acc,
                'model_state_dict': student_model.state_dict(), 'kd_strategy': args.kd_strategy,
                'student_model': args.student_model, 'seed': args.seed, 'dataset': args.dataset,
            }, exp.checkpoints_dir / 'last_model.pth')
        return best_acc

    if teacher_model is not None:
        teacher_model.eval()
    for tm in (teacher_models or []):
        tm.eval()

    run_start = time.time()
    budget = (args.max_runtime_hours * 3600.0
              if getattr(args, 'max_runtime_hours', 0) and args.max_runtime_hours > 0 else None)
    last_epoch_seconds = 0.0
    for epoch in range(start_epoch, args.epoch):
        if budget is not None and (time.time() - run_start) + 1.2 * last_epoch_seconds > budget:
            log_string('>> Wall-clock budget (%.2fh) reached; stopping cleanly at epoch %d/%d. '
                       'Re-run to resume.' % (args.max_runtime_hours, epoch, args.epoch))
            return best_acc
        log_string('Epoch %d/%d:' % (epoch + 1, args.epoch))
        epoch_start = time.time()
        student_model.train()
        train_loss = 0.0
        n_batches = 0
        for points, target in train_loader:
            if points.size(0) <= 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            points = points.to(device, non_blocking=True)  # augmented in the loader worker
            target = target.to(device).long()
            target_for_loss = None if args.unlabeled else target
            loss = _forward_loss(args, criterion, student_model, teacher_model, teacher_models,
                                 points, target_for_loss, device, amp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item(); n_batches += 1

        scheduler.step()
        train_loss /= max(1, n_batches)
        epoch_seconds = time.time() - epoch_start
        log_string('Train loss: %.4f (%.1fs)' % (train_loss, epoch_seconds))
        exp.record_metrics(split='train', epoch=epoch + 1, loss=train_loss, epoch_seconds=epoch_seconds)

        res = engine.evaluate(student_model, test_loader, num_class, device, amp=False)  # fp32 evaluation
        instance_acc, class_acc = res['instance_acc'], res['class_acc']
        log_string('Test instance %.4f, class %.4f' % (instance_acc, class_acc))
        exp.record_metrics(split='eval', epoch=epoch + 1, instance_acc=instance_acc,
                           class_acc=class_acc, best_instance_acc=best_acc)

        if instance_acc > best_acc:
            best_acc, best_class_acc, best_epoch = instance_acc, class_acc, epoch + 1
            engine.atomic_save({
                'epoch': best_epoch, 'instance_acc': instance_acc, 'class_acc': class_acc,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'kd_strategy': args.kd_strategy,
                'student_model': args.student_model, 'use_hard_labels': args.use_hard_labels,
                'unlabeled': args.unlabeled, 'seed': args.seed, 'dataset': args.dataset,
            }, exp.checkpoints_dir / 'best_model.pth')
        log_string('Best instance %.4f, class %.4f (epoch %d)' % (best_acc, best_class_acc, best_epoch))

        engine.atomic_save({
            'epoch': epoch + 1, 'student': student_model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'criterion': (criterion.state_dict() if extra else {}),
            'best_acc': best_acc, 'best_class_acc': best_class_acc, 'best_epoch': best_epoch,
            'rng': engine.get_rng_state(),
        }, resume_path)
        last_epoch_seconds = time.time() - epoch_start

    engine.atomic_save({
        'epoch': args.epoch, 'instance_acc': instance_acc, 'class_acc': class_acc,
        'model_state_dict': student_model.state_dict(), 'kd_strategy': args.kd_strategy,
        'student_model': args.student_model, 'seed': args.seed, 'dataset': args.dataset,
    }, exp.checkpoints_dir / 'last_model.pth')

    student_size = get_model_size(student_model)
    student_time = measure_inference_time(student_model, test_loader, device)
    summary = dict(best_instance_acc=best_acc, best_class_acc=best_class_acc,
                   last_instance_acc=instance_acc, last_class_acc=class_acc,
                   student_size_mb=student_size, student_inference_ms=student_time,
                   num_classes=num_class)
    if teacher_model is not None:
        t_size = get_model_size(teacher_model)
        t_time = measure_inference_time(teacher_model, test_loader, device)
        summary.update(teacher_size_mb=t_size, teacher_inference_ms=t_time,
                       size_reduction_percent=(t_size - student_size) / t_size * 100 if t_size else 0.0,
                       time_reduction_percent=(t_time - student_time) / t_time * 100 if t_time else 0.0)
    exp.record_metrics(split='summary', **summary)
    log_string('Student %.2f MB, %.2f ms' % (student_size, student_time))
    return best_acc


def main(args):
    engine.set_global_seed(args.seed)
    engine.setup_perf(use_cpu=args.use_cpu)
    amp = engine.amp_default(args)
    device = engine.pick_device(args.use_cpu)

    train_dataset, test_dataset, num_class, _ = create_classification_datasets(args)
    num_class = args.num_category or num_class
    num_workers = engine.resolve_num_workers(args)
    train_loader = engine.build_loader(train_dataset, args.batch_size, True, num_workers,
                                       drop_last=True, augment=True)
    test_loader = engine.build_loader(test_dataset, args.batch_size, False, num_workers)

    teacher_model = None
    teacher_models = None
    if args.kd_strategy == 'multi_teacher':
        if not args.teacher_paths or len(args.teacher_paths) < 2:
            raise ValueError('multi_teacher requires >=2 --teacher_paths')
        teacher_models = []
        for tp in args.teacher_paths:
            tm = importlib.import_module(args.teacher_model).get_model(num_class, normal_channel=args.use_normals).to(device)
            tm.load_state_dict(engine.safe_load(tp, map_location=device)['model_state_dict'])
            teacher_models.append(tm.eval())
    else:
        teacher_model = importlib.import_module(args.teacher_model).get_model(num_class, normal_channel=args.use_normals).to(device)
        if args.teacher_path:
            teacher_model.load_state_dict(engine.safe_load(args.teacher_path, map_location=device)['model_state_dict'])
            print(f'Loaded teacher from {args.teacher_path}')
        else:
            print('WARNING: no --teacher_path; using a randomly initialised teacher.')
        teacher_model.eval()

    if args.temperature_sweep:
        base_log = args.log_dir or 'kd_run'
        for T in args.temperature_sweep:
            a = copy.deepcopy(args)
            a.temperature = float(T)
            a.log_dir = f'{base_log}_T{T}'
            a.temperature_sweep = None
            print(f'\n=== Temperature {T} ===')
            train_one(a, device, amp, num_class, train_loader, test_loader, teacher_model, teacher_models)
        return

    train_one(args, device, amp, num_class, train_loader, test_loader, teacher_model, teacher_models)


if __name__ == '__main__':
    args = parse_args()
    if args.batch_size < 2:
        print('WARNING: Batch size raised to 2 for BatchNorm.')
        args.batch_size = 2
    main(args)
