import argparse
import logging
import sys
import os

import importlib
import numpy as np

import torch.utils.data

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp

logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.join(os.getcwd()))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--start', type=float, default=0.0, help='Start of the dataset split')
    parser.add_argument('--end', type=float, default=1.0, help='End of the dataset split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--out_dir', type=str, help='Output directory of eval results')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    test_dataset = getattr(importlib.import_module(args.dataset), 'Dataset')(args.dataset_path,
                                                                             dep=args.use_depth, rgb=args.use_rgb,
                                                                             start=args.start, end=args.end)

    # Load Network
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    net = torch.load(args.network, map_location=device)

    # Load Dataset
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    print(test_data)

    results = {'correct': 0, 'failed': 0}

    if args.out_dir:
        out_results = np.zeros(len(test_data))

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])
            if args.out_dir:
                np.save(os.path.join(args.out_dir, f'{idx}_q.npy'), q_img)
                np.save(os.path.join(args.out_dir, f'{idx}_a.npy'), ang_img)
                np.save(os.path.join(args.out_dir, f'{idx}_w.npy'), width_img)
            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_grasps(idx),
                                                   no_grasps=args.n_grasps,
                                                   grasp_width=width_img,
                                                   )
                if args.out_dir:
                    if s:
                        out_results[idx] = 1
                    np.save(os.path.join(args.out_dir, 'results.npy'), out_results)
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))
