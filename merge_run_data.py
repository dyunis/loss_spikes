import argparse
from collections import defaultdict
from glob import glob
import itertools
import json
import os

import numpy as np


# compute avg run plots over seeds for different hyp combos run
# reading data from all of {sweep_name}/{run_name}/{plots}
# so save as {sweep_name}/{hyp_name}_{value}/{avg plots over seed}
# need to read in all csvs and config dicts, group runs, then average
# values in the config dicts and remake the plots saving in the new directory
# there may be a mismatch in timesteps for different runs, so average where
# they match, and take the union where they don't
def main(config):
    config.group_dir = config.group_dir.rstrip('/')  # remove trailing slashes
    group_name = os.path.basename(config.group_dir)
    config_paths = glob(os.path.join(config.group_dir, f'{group_name}-*', 'config.json'))
    npz_paths = glob(os.path.join(config.group_dir, f'{group_name}-*', 'data.npz'))
    config_paths = sorted(config_paths)
    npz_paths = sorted(npz_paths)
    assert len(config_paths) > 0, f'no config paths globbed in {config.group_dir}'
    assert len(config_paths) == len(npz_paths), f'configs {config_paths} do not correspond to npzs {npz_paths}'
    for p1, p2 in zip(config_paths, npz_paths):
        run_name1 = os.path.basename(os.path.dirname(p1))
        run_name2 = os.path.basename(os.path.dirname(p2))
        assert run_name1 == run_name2, f'run names {run_name1} and {run_name2} are different'

    # group runs by whether they differ only by a seed
    config_dicts = []
    for p in config_paths:
        with open(p, 'r') as f:
            config_dicts.append(json.load(f))

    # get all unique hyp values that aren't seed, assumes sweep is over one hyp besides seed
    all_hyps = {k: list(set([config_dict[k] for config_dict in config_dicts])) for k in config_dicts[0]}
    swept_hyps = [k for k in all_hyps if len(all_hyps[k]) > 1 and 'seed' not in k and 'wandb' not in k and 'savedir' not in k]
    swept_hyps = sorted(swept_hyps)
    if len(swept_hyps) > 0:
        # sweep hyps besides seed
        hyp_combos = list(itertools.product(*[all_hyps[k] for k in swept_hyps]))
        hyp_combos = [{k: v for k, v in zip(swept_hyps, combo)} for combo in hyp_combos]
    else:
        # no non-seed hyps swept, just pick some constant hyp
        hyp_combos = [{'lr': all_hyps['lr'][0]}]

    # for every hyp combo
    hyp_dict = {
        'rank': {},
        'train/loss': {},
        'val/loss': {}
    }
    for combo in hyp_combos:
        # find all configs matching that combo
        combo_npzs = []
        for d, p, cp in zip(config_dicts, npz_paths, config_paths):
            use_config = True
            for c in combo: 
                if combo[c] != d[c]:
                    use_config = False
                    break
            if use_config:
                combo_npzs.append(p)
                config_path = cp
            else:
                continue

        if len(combo_npzs) <= 0:
            continue

        # collect all data matching hyp combo
        data_dict = defaultdict(lambda: [])
        for p in combo_npzs:
            # read in respective npzs matching config
            c_dict = dict(np.load(p))
            for k in c_dict:
                data_dict[k].append(c_dict[k])

        # for avg effective rank of model create giant list of all rank keys
        rank_keys = [k for k in data_dict if k.endswith('eff_rank')]
        for k in rank_keys:
            data_dict['eff_rank'].extend(data_dict[k])

        # for dim_avg sv(a), create giant list of all dim keys
        dims = list(set([c_dict[k].shape[1] for k in c_dict if k.endswith('_sv')]))
        for d in dims:
            dim_keys = [k.rstrip('sv') for k in c_dict if k.endswith('_sv') and c_dict[k].shape[1] == d]
            # get full list
            for ext in ('sv', 'sva', 'eff_rank'):
                for k in dim_keys:
                    data_dict[f'dim_{d}_{ext}'].extend(data_dict[k + ext])

        # TODO: inter layer alignment (dim avg?)
        # get alignment score for every layer
        # get average alignment score across all layers
        align_keys = [k for k in data_dict if k.endswith('align')]
        align_shapes = list(set([c_dict[k].shape[-2:] for k in align_keys]))
        for k in align_keys:
            scores = [inter_layer_alignment_score(al) for al in data_dict[k]]
            data_dict[f'{k}_score'].extend(scores)
            data_dict[f'align_score'].extend(scores)

        for shape in align_shapes:
            shape_keys = [k for k in align_keys if c_dict[k].shape[-2:] == shape]
            shape_str = f'{shape[0]}_{shape[1]}'
            for k in shape_keys:
                data_dict[f'shape_{shape_str}_align'].extend(data_dict[k])
                data_dict[f'shape_{shape_str}_align_score'].extend(data_dict[k+'_score'])

        data_dict = {k: np.array(v) for k, v in data_dict.items()}
        mean_dict = {k: np.mean(v, axis=0) for k, v in data_dict.items()}
        std_dict = {k: np.std(v, axis=0) for k, v in data_dict.items()}

        # turn combo into string name
        combo_name = [f'{k}_{v}' for k, v in combo.items()]
        combo_name = '-'.join(combo_name)
        savedir = os.path.join(config.group_dir, combo_name)
        os.makedirs(savedir, exist_ok=True)
        # save mean/std dict for hyp under new hyp dir
        np.savez(os.path.join(savedir, 'mean.npz'), **mean_dict)
        np.savez(os.path.join(savedir, 'std.npz'), **std_dict)

        rank_avg, rank_std = mean_dict['eff_rank'], std_dict['eff_rank']

        vloss_avg, vloss_std = mean_dict['val/loss'], std_dict['val/loss']
        best_epoch = np.argmin(vloss_avg)
        hyp_dict['rank'][combo_name] = (float(rank_avg[best_epoch]), float(rank_std[best_epoch]))
        hyp_dict['train/loss'][combo_name] = (float(mean_dict['train/loss'][best_epoch]), float(std_dict['train/loss'][best_epoch]))
        hyp_dict['val/loss'][combo_name] = (float(mean_dict['val/loss'][best_epoch]), float(std_dict['val/loss'][best_epoch]))

    # save hyp_dict json under group_dir
    with open(os.path.join(config.group_dir, 'hyp_data.json'), 'w') as f:
        json.dump(hyp_dict, f, indent=2)


# TODO: is there a tighter bound?
# something like 1 / sqrt(n) for each projection?
def inter_layer_alignment_score(alignment, normalize=False):
    # distance to identity matrix
    identity = np.eye(*alignment.shape[-2:])
    for _ in range(len(alignment.shape[:-2])):
        identity = identity[None]
    dists = alignment - identity
    # worst case distance between two elements is 1, so worst case we have n * m sq distance
    # stochastic matrix has a maximum of 2 * num diagonal entries l1 distance
    dists = np.sqrt((dists ** 2).sum(axis=(-2, -1)) / (alignment.shape[-2] * alignment.shape[-1]))
    # average over spatial positions for conv layers
    if len(dists.shape) == 3:
        dists = dists.mean(axis=(1, 2))
    return dists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group_dir', type=str, help='group dir of experiments that were run and evaluated')
    config = parser.parse_args()
    main(config)
