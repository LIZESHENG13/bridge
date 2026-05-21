# coding: utf-8

"""Main entry point for BRIDGE."""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BRIDGE', help='model name')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='dataset name')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    parser.add_argument('--epochs', type=int, default=None, help='override max training epochs')
    parser.add_argument('--stopping_step', type=int, default=None, help='override early stopping steps')
    parser.add_argument('--log_file_name', type=str, default=None, help='fixed log/checkpoint name')
    parser.add_argument('--use_behavior_proto', type=str2bool, default=None)
    parser.add_argument('--use_behavior_correction', type=str2bool, default=None)
    parser.add_argument('--use_private_residual', type=str2bool, default=None)
    parser.add_argument('--use_topk_correction', type=str2bool, default=None)
    parser.add_argument('--use_behavior_gate', type=str2bool, default=None)
    parser.add_argument('--use_calibration_gate', type=str2bool, default=None)
    parser.add_argument('--use_residual_gate', type=str2bool, default=None)
    parser.add_argument('--behavior_weight', type=float, default=None)
    parser.add_argument('--behavior_topk', type=int, default=None)
    parser.add_argument('--behavior_eval_topk', type=int, default=None)
    parser.add_argument('--train_candidate_aware', type=str2bool, default=None)
    parser.add_argument('--train_candidate_topk', type=int, default=None)
    parser.add_argument('--correction_scope', type=str, choices=['topk', 'global'], default=None)
    parser.add_argument('--behavior_score_norm', type=str, default=None)
    parser.add_argument('--private_residual_weight', type=float, default=None)
    parser.add_argument('--aux_base_weight', type=float, default=None)
    parser.add_argument('--aux_high_weight', type=float, default=None)
    parser.add_argument('--num_freq_bands', type=int, default=None)
    parser.add_argument(
        '--freq_decomp_method',
        type=str,
        choices=['svd', 'gram', 'dct', 'random', 'none_equal_capacity', 'none', 'no_decomp_equal_capacity'],
        default=None,
    )
    parser.add_argument('--reg_weight', type=float, default=None)
    parser.add_argument('--ib_weight', type=float, default=None)
    parser.add_argument('--feature_ablation', type=str, choices=['all', 'no_image', 'no_text', 'no_content'], default=None)
    parser.add_argument('--use_item_graph', type=str2bool, default=None)
    parser.add_argument('--behavior_proto_weight', type=float, default=None)
    parser.add_argument('--behavior_proto_loss_weight', type=float, default=None)

    args, _ = parser.parse_known_args()
    config_dict = {
        'gpu_id': args.gpu_id,
        'seed': [args.seed],
        'log_file_name': args.log_file_name,
    }
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    if args.stopping_step is not None:
        config_dict['stopping_step'] = args.stopping_step
    for key in (
        'use_behavior_proto',
        'use_behavior_correction',
        'use_private_residual',
        'use_topk_correction',
        'use_behavior_gate',
        'use_calibration_gate',
        'use_residual_gate',
        'behavior_weight',
        'behavior_topk',
        'behavior_eval_topk',
        'train_candidate_aware',
        'train_candidate_topk',
        'correction_scope',
        'behavior_score_norm',
        'private_residual_weight',
        'aux_base_weight',
        'aux_high_weight',
        'num_freq_bands',
        'freq_decomp_method',
        'reg_weight',
        'ib_weight',
        'feature_ablation',
        'use_item_graph',
        'behavior_proto_weight',
        'behavior_proto_loss_weight',
    ):
        value = getattr(args, key)
        if value is not None:
            config_dict[key] = [value]

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
