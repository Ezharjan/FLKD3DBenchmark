"""Centralised PointNet/PointNet++ classification baseline.

Uses the shared engine: fp32 by default (bf16 AMP auto-enabled on bf16-capable
GPUs), multi-worker data loading, confusion-matrix evaluation, dataset-namespaced
outputs and per-epoch auto-resume.  (Original author: Benny, Nov 2019.)
"""

import os
import sys
import time
import argparse
import importlib
from pathlib import Path

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR)

from data_utils.dataset_factory import create_classification_datasets  # noqa: E402
from flkd import engine  # noqa: E402
from flkd.engine import augment_points, autocast_ctx  # noqa: E402
from flkd.kd_utils import get_model_size, measure_inference_time  # noqa: E402
from experiment import prepare_experiment  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--no_amp', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--model', default='pointnet2_cls_ssg')
    parser.add_argument('--num_category', default=None, type=int)
    parser.add_argument('--dataset', type=str, default='modelnet40')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_root', type=str, default='outputs')
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--cache_dataset', dest='cache_dataset', action='store_true', default=True)
    parser.add_argument('--no_cache', dest='cache_dataset', action='store_false')
    parser.add_argument('--min_class_samples', type=int, default=1)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--use_normals', action='store_true', default=False)
    parser.add_argument('--process_data', action='store_true', default=False)
    parser.add_argument('--use_uniform_sample', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_resume', dest='resume', action='store_false', default=True)
    parser.add_argument('--max_runtime_hours', type=float, default=0.0,
                        help='graceful wall-clock budget in hours (0 = unlimited); the trainer '
                             'finishes the current epoch, checkpoints, and exits cleanly before '
                             'this budget so a time-limited job resumes with no lost work')
    return parser.parse_args()


def inplace_relu(m):
    if m.__class__.__name__.find('ReLU') != -1:
        m.inplace = True


def main(args):
    engine.set_global_seed(args.seed)
    engine.setup_perf(use_cpu=args.use_cpu)
    amp = engine.amp_default(args)
    device = engine.pick_device(args.use_cpu)

    exp = prepare_experiment(subdirs=[args.dataset, 'classification'], log_dir=args.log_dir,
                             name_hint=args.model, output_root=Path(args.output_root),
                             logger_name_prefix='classification')
    logger = exp.logger

    def log_string(m):
        logger.info(m); print(m)

    exp.log_args(args)
    log_string(f'Device {device}; AMP(bf16)={amp}')

    train_dataset, test_dataset, num_class, _ = create_classification_datasets(args)
    num_class = args.num_category or num_class
    num_workers = engine.resolve_num_workers(args)
    train_loader = engine.build_loader(train_dataset, args.batch_size, True, num_workers,
                                       drop_last=True, augment=True)
    test_loader = engine.build_loader(test_dataset, args.batch_size, False, num_workers)

    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class, normal_channel=args.use_normals).to(device)
    criterion = model.get_loss().to(device)
    classifier.apply(inplace_relu)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_instance_acc = best_class_acc = 0.0
    best_epoch = 0
    start_epoch = 0
    instance_acc = class_acc = 0.0

    resume_path = exp.checkpoints_dir / 'resume.pth'
    if args.resume and resume_path.exists():
        try:
            st = engine.safe_load(resume_path, map_location=device)
            start_epoch = st['epoch']
            classifier.load_state_dict(st['model'])
            optimizer.load_state_dict(st['optimizer'])
            scheduler.load_state_dict(st['scheduler'])
            best_instance_acc = st['best_instance_acc']; best_class_acc = st['best_class_acc']
            best_epoch = st['best_epoch']
            engine.set_rng_state(st.get('rng'))
            log_string(f'>> Resumed from epoch {start_epoch}/{args.epoch} (best {best_instance_acc:.4f}).')
        except Exception as exc:
            log_string(f'>> Resume failed ({exc}); starting fresh.')
            start_epoch = 0

    if start_epoch >= args.epoch:
        log_string('Training already complete; nothing to do.')
        if not (exp.checkpoints_dir / 'last_model.pth').exists():
            engine.atomic_save({
                'epoch': args.epoch, 'instance_acc': best_instance_acc, 'class_acc': best_class_acc,
                'model_state_dict': classifier.state_dict(), 'seed': args.seed, 'dataset': args.dataset,
            }, exp.checkpoints_dir / 'last_model.pth')
        return

    log_string('Start training...')
    run_start = time.time()
    budget = args.max_runtime_hours * 3600.0 if args.max_runtime_hours and args.max_runtime_hours > 0 else None
    last_epoch_seconds = 0.0
    for epoch in range(start_epoch, args.epoch):
        if budget is not None and (time.time() - run_start) + 1.2 * last_epoch_seconds > budget:
            log_string('>> Wall-clock budget (%.2fh) reached; stopping cleanly at epoch %d/%d. '
                       'Re-run to resume.' % (args.max_runtime_hours, epoch, args.epoch))
            return
        epoch_t0 = time.time()
        log_string('Epoch %d/%d:' % (epoch + 1, args.epoch))
        classifier.train()
        correct_sum = 0
        seen = 0
        for points, target in train_loader:
            if points.size(0) <= 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            points = points.to(device, non_blocking=True)  # augmented in the loader worker
            target = target.to(device).long()
            with autocast_ctx(device, amp):
                pred, trans_feat = classifier(points)
                loss = criterion(pred, target, trans_feat)
            loss.backward()
            optimizer.step()
            correct_sum += pred.argmax(1).eq(target).sum().item()
            seen += target.numel()
        scheduler.step()
        train_acc = correct_sum / max(1, seen)
        log_string('Train Instance Accuracy: %.4f' % train_acc)
        exp.record_metrics(split='train', epoch=epoch + 1, instance_acc=train_acc)

        res = engine.evaluate(classifier, test_loader, num_class, device, amp=False)  # fp32 evaluation
        instance_acc, class_acc = res['instance_acc'], res['class_acc']
        is_best = instance_acc >= best_instance_acc
        if is_best:
            best_instance_acc, best_epoch = instance_acc, epoch + 1
        best_class_acc = max(best_class_acc, class_acc)
        log_string('Test Instance %.4f, Class %.4f | Best Instance %.4f, Class %.4f'
                   % (instance_acc, class_acc, best_instance_acc, best_class_acc))
        exp.record_metrics(split='eval', epoch=epoch + 1, instance_acc=instance_acc,
                           class_acc=class_acc, best_instance_acc=best_instance_acc,
                           best_class_acc=best_class_acc)

        if is_best:
            engine.atomic_save({
                'epoch': best_epoch, 'instance_acc': instance_acc, 'class_acc': class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'seed': args.seed, 'dataset': args.dataset,
            }, exp.checkpoints_dir / 'best_model.pth')

        engine.atomic_save({
            'epoch': epoch + 1, 'model': classifier.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_instance_acc': best_instance_acc, 'best_class_acc': best_class_acc,
            'best_epoch': best_epoch, 'rng': engine.get_rng_state(),
        }, resume_path)
        last_epoch_seconds = time.time() - epoch_t0

    engine.atomic_save({
        'epoch': args.epoch, 'instance_acc': instance_acc, 'class_acc': class_acc,
        'model_state_dict': classifier.state_dict(), 'seed': args.seed, 'dataset': args.dataset,
    }, exp.checkpoints_dir / 'last_model.pth')
    size_mb = get_model_size(classifier)
    infer_ms = measure_inference_time(classifier, test_loader, device)
    exp.record_metrics(split='summary', best_instance_acc=best_instance_acc,
                       best_class_acc=best_class_acc, last_instance_acc=instance_acc,
                       last_class_acc=class_acc, model_size_mb=size_mb,
                       inference_time_ms=infer_ms, num_classes=num_class)
    log_string('End of training. Best %.4f at epoch %d.' % (best_instance_acc, best_epoch))


if __name__ == '__main__':
    args = parse_args()
    if args.batch_size < 2:
        args.batch_size = 2
    main(args)
