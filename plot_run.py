import argparse
import os
from glob import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main(metrics_dir):
    metrics_dir = config.metrics_dir
    savedir = '-'.join(os.path.normpath(metrics_dir).split(os.sep)[-2:])
    savedir = os.path.join('plots', savedir)
    os.makedirs(savedir, exist_ok=True)
    mean_dict = dict(np.load(os.path.join(metrics_dir, 'mean.npz')))
    std_dict = dict(np.load(os.path.join(metrics_dir, 'std.npz')))

    splits = ['train', 'val']
    keys = ['loss', 'err', 'cer', 'wer']
    for split in splits:
        for key in keys:
            if f'{split}/{key}' not in mean_dict:
                continue
            plot_keys = [f'{split}/{key}', f'{split}/pruned_bot_{key}', f'{split}/pruned_top_{key}']
            colors = matplotlib.colormaps['Set1'].colors
            fig, ax = plt.subplots()
            for k, color in zip(plot_keys, colors):
                plot_mean_std(savedir, mean_dict, std_dict, k, ax=ax, color=color)
            ax.legend()
            fig.savefig(os.path.join(savedir, f'{split}-{key}.png'))
            plt.close(fig)

    # plot eff rank
    rank_keys = [k for k in mean_dict if 'eff_rank' in k]
    for k in rank_keys:
        plot_mean_std(savedir, mean_dict, std_dict, k)

    # plot skew and diffskew
    skew_keys = [k for k in mean_dict if 'skew' in k]
    for k in skew_keys:
        plot_mean_std(savedir, mean_dict, std_dict, k)

    # plot of svs for each layer
    sv_keys = [k for k in mean_dict if 'sv' in k and 'sva' not in k]
    for k in sv_keys:
        plot_svs(savedir, mean_dict, k)

    # plot of sva for each layer over time
    # - individual
    # - dim avg
    # come up with metric to show top component stability based on SVA?
    sva_keys = [k for k in mean_dict if 'sva' in k]
    for k in sva_keys:
        sva = mean_dict[k]
        plot_sva_diag(savedir, k, sva)
        steps = len(sva)
        for step in range(steps):
            plot_sva_step(savedir, k, sva, step)

    # plot inter-layer alignment
    align_keys = [k for k in mean_dict.keys() if k.endswith('align')]
    # align_score_keys = [k for k in mean_dict.keys() if k.endswith('align_score')]

    for k in align_keys:
        sva = mean_dict[k]
        if len(sva.shape) == 5:
            # for conv layers we have spatial dimension
            sva = sva[:, sva.shape[1]//2, sva.shape[2]//2]  # center position
            # sva = np.mean(sva, axis=(1, 2))  # mean over spatial positions
            # sva = sva[:, 0, 0]  # corner
            # sva = sva[:, 0, 2]  # corner
        elif len(sva.shape) == 4:
            # for lstm layers we have 4 separate matrices in each param
            # output mat sends the strongest signal as this is directly connected to next layer
            # TODO: double check computing the alignment correctly for lstm
            # sva = np.mean(sva, axis=1)  # mean over forget, input, output, cell matrices
            # sva = sva[:, 0]  # forget mat
            # sva = sva[:, 1]  # input mat
            sva = sva[:, 2]  # output mat
            # sva = sva[:, 3]  # cell mat
        plot_align_diag(savedir, k, sva, rank=100)
        early_steps = list(range(5))
        late_steps = list(range(len(sva)))[5:]
        late_steps = [ls for i, ls in enumerate(late_steps) if i%10 == 0]
        for step in [*early_steps, *late_steps]:
            plot_align_step(savedir, k, sva, step)


# plot of eval metrics
def plot_mean_std(savedir, mean_dict, std_dict, key, ax=None, color='blue'):
    save = False
    if ax is None:
        save = True
        fig, ax = plt.subplots()
    mean, std = mean_dict[key], std_dict[key]
    x = np.arange(len(mean))
    ax.plot(x, mean, color=color, label=key, linewidth=2)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
    if save:
        savepath = os.path.join(savedir, 'metrics', f'{key.replace("/", "-")}.png')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath)
        plt.close(fig)


def plot_svs(savedir, mean_dict, key):
    svs = mean_dict[key]
    fig, ax = plt.subplots()
    num_svs = svs.shape[1]
    colormap = matplotlib.colormaps['viridis'].resampled(num_svs)
    for i in range(num_svs):
        ax.plot(svs[:, i], color=colormap(i))
    savepath = os.path.join(savedir, 'sv', f'{key.replace("/", "-")}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    plt.close(fig)


# plot diagonal heatmap
def plot_sva_diag(savedir, key, sva, rank=100):
    sva = sva[:, :rank, :rank]
    diags = []
    for i in range(len(sva)):
        diags.append(np.diag(sva[i]))
    diags = np.array(diags)
    diags = diags.T
    # plot heatmap of sva, colorbar from 0-1
    fig, ax = plt.subplots()
    img = ax.imshow(diags, cmap='inferno', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    savepath = os.path.join(savedir, 'sva', key, 'diag.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    plt.close(fig)


# plot full heatmap at individual step
def plot_sva_step(savedir, key, sva, step, rank=100):
    sva = sva[:, :rank, :rank]
    fig, ax = plt.subplots()
    img = ax.imshow(sva[step], cmap='inferno', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    savepath = os.path.join(savedir, 'sva', key, f'step_{step}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    plt.close(fig)


# plot diagonal heatmap
def plot_align_diag(savedir, key, sva, rank=100):
    sva = sva[:, :rank, :rank]
    diags = []
    for i in range(len(sva)):
        diags.append(np.diag(sva[i]))
    diags = np.array(diags)
    diags = diags.T
    # plot heatmap of sva, colorbar from 0-1
    fig, ax = plt.subplots()
    img = ax.imshow(diags, cmap='inferno', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    savepath = os.path.join(savedir, 'align', key, 'diag.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    plt.close(fig)


# plot full heatmap at individual step
def plot_align_step(savedir, key, sva, step, rank=100):
    sva = sva[:, :rank, :rank]
    fig, ax = plt.subplots()
    img = ax.imshow(sva[step], cmap='inferno', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    savepath = os.path.join(savedir, 'align', key, f'step_{step}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_dir', type=str, default=None, help='metrics directory to load mean,std data from')
    config = parser.parse_args()
    # config.metrics_dir = '/share/data/speech/Data/dyunis/exps/wandb/lmc_svd/imgclass_resnet_cifar10/lr_0.1'
    # config.metrics_dir = '/share/data/speech/Data/dyunis/exps/wandb/lmc_svd/speech_lstm_libri/lr_0.0003'
    assert os.path.exists(config.metrics_dir)
    main(config)
