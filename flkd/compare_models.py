"""
Compare Original, Federated Learning, and Knowledge Distillation Models
"""

import os
import sys
import torch
import numpy as np
import argparse
import importlib
from pathlib import Path
from tqdm import tqdm

OUTPUT_ROOT = Path('./outputs')

# Fix matplotlib backend for Windows and headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display and PIL issues
import matplotlib.pyplot as plt

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = BASE_DIR
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Import local modules
from data_utils.dataset_factory import create_classification_datasets
from flkd import engine
from flkd.kd_utils import (
    get_model_size,
    measure_inference_time,
    SmallPointNetCls,
    SmallPointNet2ClsSsg,
    DGCNNClsStudent,
)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Compare Models')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--num_category', default=None, type=int, help='number of classes (overrides dataset-derived count)')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset name (modelnet40, modelnet10, omni_object3d, ycb, gazebosim, craniosynostosis)')
    parser.add_argument('--data_root', type=str, default='data', help='root directory that contains dataset folders')
    parser.add_argument('--train_split', type=float, default=0.8, help='train split ratio for folder-based datasets (match training)')
    parser.add_argument('--dataset_seed', type=int, default=0, help='random seed for dataset train/test split (match training)')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    # Model paths
    parser.add_argument('--original_model_path', type=str, required=True, help='path to original model')
    parser.add_argument('--federated_model_path', type=str, required=True, help='path to federated learning model')
    parser.add_argument('--distilled_model_path', type=str, required=True, help='path to knowledge distillation model')
    parser.add_argument('--original_model', default='pointnet2_cls_ssg', help='original model name')
    parser.add_argument('--student_model', default='small_pointnet2',
                        choices=['small_pointnet', 'small_pointnet2', 'dgcnn'],
                        help='student model type (small_pointnet/small_pointnet2 are same family; dgcnn is heterogeneous)')

    # Output
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_ROOT / 'results'), help='output directory for results')

    return parser.parse_args()

def test(model, loader, num_class=40, device=None):
    """Evaluate via the shared confusion-matrix evaluator (``engine.evaluate``).

    Uses the same protocol as the training scripts so the Original / Federated /
    Distilled models are scored the same way: one fp32 forward pass per sample,
    overall accuracy (OA) and macro per-class accuracy (mAcc) from a full
    confusion matrix (independent of batch size).  Returns
    ``(instance_acc, class_acc)``.
    """
    res = engine.evaluate(model, loader, num_class, device, amp=False)
    return res['instance_acc'], res['class_acc']

def main(args):
    '''SET DEVICE'''
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    '''DATA LOADING'''
    print('Load dataset ...')
    _, test_dataset, num_class, class_names = create_classification_datasets(args)

    # Use fewer workers on Windows to avoid shared memory issues
    num_workers = 0 if os.name == 'nt' else 2

    testDataLoader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False if num_workers == 0 else True
    )

    '''MODEL LOADING'''
    num_class = args.num_category or num_class

    # Load original model
    print('Loading original model...')
    original_model_module = importlib.import_module(args.original_model)
    original_model = original_model_module.get_model(num_class, normal_channel=args.use_normals).to(device)
    checkpoint = torch.load(args.original_model_path, weights_only=False)
    original_model.load_state_dict(checkpoint['model_state_dict'])

    # Load federated model
    print('Loading federated model...')
    federated_model = original_model_module.get_model(num_class, normal_channel=args.use_normals).to(device)
    checkpoint = torch.load(args.federated_model_path, weights_only=False)
    federated_model.load_state_dict(checkpoint['model_state_dict'])

    # Load distilled model
    print('Loading distilled model...')
    if args.student_model == 'small_pointnet':
        distilled_model = SmallPointNetCls(num_class, normal_channel=args.use_normals).to(device)
    elif args.student_model == 'dgcnn':
        distilled_model = DGCNNClsStudent(num_class, normal_channel=args.use_normals).to(device)
    else:  # small_pointnet2
        distilled_model = SmallPointNet2ClsSsg(num_class, normal_channel=args.use_normals).to(device)
    checkpoint = torch.load(args.distilled_model_path, weights_only=False)
    distilled_model.load_state_dict(checkpoint['model_state_dict'])

    '''EVALUATION'''
    print('Evaluating models...')

    # Test accuracy
    original_acc, original_class_acc = test(original_model, testDataLoader, num_class=num_class, device=device)
    federated_acc, federated_class_acc = test(federated_model, testDataLoader, num_class=num_class, device=device)
    distilled_acc, distilled_class_acc = test(distilled_model, testDataLoader, num_class=num_class, device=device)

    print('Original Model - Instance Accuracy: %f, Class Accuracy: %f' % (original_acc, original_class_acc))
    print('Federated Model - Instance Accuracy: %f, Class Accuracy: %f' % (federated_acc, federated_class_acc))
    print('Distilled Model - Instance Accuracy: %f, Class Accuracy: %f' % (distilled_acc, distilled_class_acc))

    # Model size
    original_size = get_model_size(original_model)
    federated_size = get_model_size(federated_model)
    distilled_size = get_model_size(distilled_model)

    print('Original Model Size: %.2f MB' % original_size)
    print('Federated Model Size: %.2f MB' % federated_size)
    print('Distilled Model Size: %.2f MB' % distilled_size)
    print('Distilled Size Reduction: %.2f%%' % ((original_size - distilled_size) / original_size * 100))

    # Inference time
    original_time = measure_inference_time(original_model, testDataLoader, device)
    federated_time = measure_inference_time(federated_model, testDataLoader, device)
    distilled_time = measure_inference_time(distilled_model, testDataLoader, device)

    print('Original Model Inference Time: %.2f ms' % original_time)
    print('Federated Model Inference Time: %.2f ms' % federated_time)
    print('Distilled Model Inference Time: %.2f ms' % distilled_time)
    print('Distilled Speed Improvement: %.2f%%' % ((original_time - distilled_time) / original_time * 100))

    '''VISUALIZATION'''
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Accuracy comparison
        plt.figure(figsize=(10, 6))
        models = ['Original', 'Federated', 'Distilled']
        instance_accs = [original_acc, federated_acc, distilled_acc]
        class_accs = [original_class_acc, federated_class_acc, distilled_class_acc]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, instance_accs, width, label='Instance Accuracy', color='steelblue')
        rects2 = ax.bar(x + width/2, class_accs, width, label='Class Accuracy', color='coral')

        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Add values on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}%'.format(height * 100),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig(str(output_dir / 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Model size comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        sizes = [original_size, federated_size, distilled_size]
        colors = ['steelblue', 'mediumseagreen', 'tomato']

        bars = ax.bar(models, sizes, color=colors)
        ax.set_ylabel('Model Size (MB)', fontsize=12)
        ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Add values on top of bars
        for i, (bar, v) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(sizes)*0.01,
                    '{:.2f} MB'.format(v), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(str(output_dir / 'size_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Inference time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        times = [original_time, federated_time, distilled_time]

        bars = ax.bar(models, times, color=colors)
        ax.set_ylabel('Inference Time (ms)', fontsize=12)
        ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Add values on top of bars
        for i, (bar, v) in enumerate(zip(bars, times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(times)*0.01,
                    '{:.2f} ms'.format(v), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(str(output_dir / 'time_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print('Saved comparison figures.')

    except Exception as e:
        print(f'Warning: Failed to create visualizations: {e}')
        print('Continuing with CSV export...')

    # Save results to CSV
    import csv
    try:
        with open(str(output_dir / 'results.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Instance Accuracy (%)', 'Class Accuracy (%)', 'Model Size (MB)', 'Inference Time (ms)'])
            writer.writerow(['Original', original_acc * 100, original_class_acc * 100, original_size, original_time])
            writer.writerow(['Federated', federated_acc * 100, federated_class_acc * 100, federated_size, federated_time])
            writer.writerow(['Distilled', distilled_acc * 100, distilled_class_acc * 100, distilled_size, distilled_time])
            writer.writerow([])  # Empty row
            writer.writerow(['Comparison Metrics', 'Value'])
            writer.writerow(['Distilled Size Reduction (%)', ((original_size - distilled_size) / original_size * 100)])
            writer.writerow(['Distilled Speed Improvement (%)', ((original_time - distilled_time) / original_time * 100)])
            writer.writerow(['Federated vs Original Accuracy Diff (%)', (federated_acc - original_acc) * 100])
            writer.writerow(['Distilled vs Original Accuracy Diff (%)', (distilled_acc - original_acc) * 100])

        print('Saved results to CSV.')
    except Exception as e:
        print(f'Warning: Failed to save CSV: {e}')

    # Print summary
    print('\n' + '='*80)
    print('COMPARISON SUMMARY')
    print('='*80)
    print(f'\nAccuracy Metrics:')
    print(f'  Original Model:   Instance={original_acc*100:.2f}%  Class={original_class_acc*100:.2f}%')
    print(f'  Federated Model:  Instance={federated_acc*100:.2f}%  Class={federated_class_acc*100:.2f}%')
    print(f'  Distilled Model:  Instance={distilled_acc*100:.2f}%  Class={distilled_class_acc*100:.2f}%')
    
    print(f'\nEfficiency Metrics:')
    print(f'  Model Size:       Original={original_size:.2f}MB  Federated={federated_size:.2f}MB  Distilled={distilled_size:.2f}MB')
    print(f'  Inference Time:   Original={original_time:.2f}ms  Federated={federated_time:.2f}ms  Distilled={distilled_time:.2f}ms')
    
    print(f'\nImprovements:')
    print(f'  Distilled Size Reduction:     {((original_size - distilled_size) / original_size * 100):.2f}%')
    print(f'  Distilled Speed Improvement:  {((original_time - distilled_time) / original_time * 100):.2f}%')
    print(f'  Federated Accuracy Change:    {(federated_acc - original_acc) * 100:+.2f}%')
    print(f'  Distilled Accuracy Change:    {(distilled_acc - original_acc) * 100:+.2f}%')
    print('='*80)

    print('\nResults saved to', args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)
