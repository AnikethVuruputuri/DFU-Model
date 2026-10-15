import argparse
import copy
from typing import Dict, Any, List

BASE_CONFIG = {
    'arch': 'full',
    'epochs': 60,
    'batch_size': 8,
    'lr_backbone': 3e-6,
    'lr_head': 1e-4,
    'dropout': 0.3,
    'img_size': 224,
    'val_split': 0.2,
    'seed': 42,
    'grad_clip': 1.0,
    'loss_pathology': 'cbfocal',
    'focal_beta': 0.999,
    'focal_gamma': 2.0,
    'lambda_path': 1.0,
    'lambda_coral': 1.0,
    'lambda_lmcot': 0.5,
    'lambda_mmd': 0.1,
    'lambda_orth': 0.1,
    'enable_wagner': True,
    'use_hard_lmcot': True,
    'monitor_metric': 'macro_f1',
    'deterministic': True,
    'early_stopping_patience': 15,
    'label_smoothing': 0.0,
    'warmup_epochs': 3,
    'weight_decay': 1e-3,
    'weight_decay_backbone': 1e-4,
    'finetune_strategy': 'none',
    'unfreeze_epoch': 5,
    'augmentation_level': 'baseline',
}

FINETUNE_EXPERIMENTS = {
    'FT1_FullFinetune': {
        'name': 'Full Fine-tuning (all parameters trainable)',
        'description': 'Baseline: all backbone + head parameters updated with differential LR',
        'diff': {}
    },
    'FT2_PartialFinetune': {
        'name': 'Partial Fine-tuning (top stages only)',
        'diff': {'partial_finetuning': True}
    },
    'FT3_GradualUnfreeze': {
        'name': 'Gradual Unfreezing (frozen -> unfreeze at epoch 5)',
        'diff': {'finetune_strategy': 'gradual_unfreeze', 'unfreeze_epoch': 5}
    },
    'FT4_HeadWarmup': {
        'name': 'Head Warmup then Full Fine-tune',
        'diff': {'finetune_strategy': 'head_warmup', 'unfreeze_epoch': 5, 'lr_head': 3e-4}
    },
}

ABLATION_EXPERIMENTS = {
    'AB1_PathOnly': {
        'name': 'Pathology Classification Only',
        'description': 'Only pathology CE/focal loss, no auxiliary tasks',
        'diff': {'lambda_coral': 0.0, 'lambda_lmcot': 0.0, 'lambda_mmd': 0.0, 'lambda_orth': 0.0, 'enable_wagner': False}
    },
    'AB2_PathCoral': {
        'name': 'Pathology + Wagner CORAL',
        'diff': {'lambda_lmcot': 0.0, 'lambda_mmd': 0.0, 'lambda_orth': 0.0}
    },
    'AB3_PathLMCoT': {
        'name': 'Pathology + LMCoT',
        'diff': {'lambda_coral': 0.0, 'lambda_mmd': 0.0, 'lambda_orth': 0.0, 'enable_wagner': False}
    },
    'AB4_PathMMD': {
        'name': 'Pathology + MMD',
        'diff': {'lambda_coral': 0.0, 'lambda_lmcot': 0.0, 'lambda_orth': 0.0}
    },
    'AB5_PathOrth': {
        'name': 'Pathology + Orthogonality',
        'diff': {'lambda_coral': 0.0, 'lambda_lmcot': 0.0, 'lambda_mmd': 0.0, 'enable_wagner': False}
    },
    'AB6_FullMultitask': {
        'name': 'Complete Multi-task Objective',
        'diff': {}
    },
}

IMBALANCE_EXPERIMENTS = {
    'IM1_CBFocal': {
        'name': 'Class-Balanced Focal Loss (existing baseline)',
        'diff': {'loss_pathology': 'cbfocal'}
    },
    'IM2_WeightedCE': {
        'name': 'Weighted Cross-Entropy',
        'diff': {'loss_pathology': 'wce'}
    },
    'IM3_WeightedSampler': {
        'name': 'Weighted Sampling + Standard CE',
        'diff': {'loss_pathology': 'ce', 'use_weighted_sampler': True, 'sampler_mode': 'inverse_freq'}
    },
    'IM4_SamplerPlusFocal': {
        'name': 'Weighted Sampling + CB-Focal (combined)',
        'diff': {'loss_pathology': 'cbfocal', 'use_weighted_sampler': True, 'sampler_mode': 'sqrt_inverse'}
    },
}

AUGMENTATION_EXPERIMENTS = {
    'AUG1_Baseline': {
        'name': 'Baseline Augmentation (current)',
        'diff': {'augmentation_level': 'baseline'}
    },
    'AUG2_Mild': {
        'name': 'Mild Augmentation',
        'diff': {'augmentation_level': 'mild'}
    },
    'AUG3_Strong': {
        'name': 'Strong Augmentation',
        'diff': {'augmentation_level': 'strong'}
    },
    'AUG4_Clinical': {
        'name': 'Clinically-Appropriate Augmentation',
        'diff': {'augmentation_level': 'clinical'}
    },
}

SMOOTHING_EXPERIMENTS = {
    'LS0_None': {'name': 'No Label Smoothing', 'diff': {'label_smoothing': 0.0}},
    'LS1_Light': {'name': 'Light Label Smoothing (0.05)', 'diff': {'label_smoothing': 0.05}},
    'LS2_Standard': {'name': 'Standard Label Smoothing (0.1)', 'diff': {'label_smoothing': 0.1}},
}

RECOMMENDED_CONFIG = {
    'name': 'Recommended Optimized Configuration',
    'description': 'Based on validation experiments. To be updated after ablation results.',
    'diff': {
        'arch': 'full',
        'epochs': 60,
        'warmup_epochs': 3,
        'early_stopping_patience': 15,
        'finetune_strategy': 'head_warmup',
        'unfreeze_epoch': 5,
        'augmentation_level': 'clinical',
        'label_smoothing': 0.05,
        'lr_backbone': 1e-5,
        'lr_head': 3e-4,
        'weight_decay': 5e-4,
        'weight_decay_backbone': 1e-4,
    }
}

ALL_GROUPS = {
    'Finetune': FINETUNE_EXPERIMENTS,
    'Ablation': ABLATION_EXPERIMENTS,
    'Imbalance': IMBALANCE_EXPERIMENTS,
    'Augmentation': AUGMENTATION_EXPERIMENTS,
    'Smoothing': SMOOTHING_EXPERIMENTS,
    'Recommended': {'REC_Optimized': RECOMMENDED_CONFIG}
}

def get_all_experiments() -> Dict[str, Any]:
    """Return a dict of all registered experiments."""
    all_exps = {}
    for group_dict in ALL_GROUPS.values():
        all_exps.update(group_dict)
    return all_exps

def resolve_config(experiment_id: str, seed: int = 42) -> Dict[str, Any]:
    """Resolve a named experiment into a complete configuration dict."""
    all_exps = get_all_experiments()
    if experiment_id not in all_exps:
        raise ValueError(f"Experiment {experiment_id} not found.")
    
    config = copy.deepcopy(BASE_CONFIG)
    diff = all_exps[experiment_id].get('diff', {})
    config.update(diff)
    config['seed'] = seed
    return config

def config_to_args(config: Dict[str, Any], output_dir: str = None) -> List[str]:
    """Convert a config dict to a list of command-line argument strings."""
    args = []
    for k, v in config.items():
        if isinstance(v, bool):
            if v:
                args.append(f'--{k}')
        else:
            args.extend([f'--{k}', str(v)])
    if output_dir:
        args.extend(['--output_dir', output_dir])
    return args

def print_experiment_table():
    """Print a formatted table of all registered experiments."""
    print(f"{'Experiment ID':<25} | {'Name':<50}")
    print("-" * 80)
    for group_name, group_dict in ALL_GROUPS.items():
        print(f"\n--- {group_name} ---")
        for exp_id, exp_info in group_dict.items():
            print(f"{exp_id:<25} | {exp_info.get('name', '')[:50]:<50}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment Configurations")
    parser.add_argument('--experiment', type=str, help="Print full config for a specific experiment")
    parser.add_argument('--generate_commands', action='store_true', help="Print training commands for all experiments")
    
    args = parser.parse_args()
    
    if args.experiment:
        try:
            config = resolve_config(args.experiment)
            print(f"Configuration for {args.experiment}:")
            for k, v in config.items():
                print(f"  {k}: {v}")
        except ValueError as e:
            print(e)
    elif args.generate_commands:
        all_exps = get_all_experiments()
        print("Training commands for all experiments:")
        for exp_id in all_exps:
            config = resolve_config(exp_id)
            arg_list = config_to_args(config, output_dir=f"results/{exp_id}")
            print(f"python train_dfu_siam.py {' '.join(arg_list)}")
    else:
        print_experiment_table()
