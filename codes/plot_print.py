#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable=invalid-name,too-many-locals,missing-docstring
"""
Graphics and tables. This is a mess and I do not recommend you to go through it.
"""
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from codes.regression import GroupAMTL
pd.options.display.float_format = '{:,.3f}'.format
cmap_div = sns.color_palette("RdBu_r", n_colors=30)
cmap = sns.light_palette("purple", reverse=False)

ZERO_PRECISION = 1e-3


def paper_hinton_bs(exp, pos, method, max_weight=None, figsize=None):
    """
    (Art) Hintonmap dos Ws de todos os métodos do experimento na posicao pos.
    Code from: https://matplotlib.org/gallery/specialty_plots/hinton_demo.html
    """
    _, axs = plt.subplots(nrows=2, ncols=len(pos)+1, figsize=figsize)
    # dataset
    Bs = exp.resul['objs'][0]['dataset'].Bs

    hinton(Bs[0].T, max_weight=max_weight, ax=axs[0, 0])
    hinton(Bs[1].T, max_weight=max_weight, ax=axs[1, 0])
    axs[0, 0].tick_params(
        axis='both',          # changes apply to both axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,         # ticks along the left edge are off
        right=False,         # ticks along the right edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False)  # labels along the bottom edge are off

    axs[0, 0].set_title('B of group 1')
    axs[1, 0].set_title('B of group 2')
    for ind, val in enumerate(pos):
        Bs = exp.resul['objs'][val][method]['resul'][0]['Bs']
        hinton(Bs[0].T, max_weight=max_weight, ax=axs[0, ind+1])
        hinton(Bs[1].T, max_weight=max_weight, ax=axs[1, ind+1])
        for i in [0, 1]:
            axs[i, ind+1].tick_params(
                axis='both',          # changes apply to both axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,         # ticks along the left edge are off
                right=False,         # ticks along the right edge are off
                labelbottom=False,  # labels along the bottom edge are off
                labelleft=False)  # labels along the bottom edge are off
    # axs[0, 1].set_title('50 samples')
    # axs[0, 2].set_title('500 samples')
    axs[1, 0].set_ylabel('from task')
    axs[1, 0].set_xlabel('to task')
    axs[1, 0].set_yticklabels([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    axs[1, 0].set_xticklabels([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # plt.tight_layout()
    # plt.savefig('/home/churros/pasta/codes/gamtl/figs/art_hinton_bs.pdf')
    plt.show()

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('white')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=9))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=9))

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if abs(w) < ZERO_PRECISION else 'gray'
        size = np.sqrt(np.abs(w) / max_weight) + 0.0001
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    ax.autoscale_view()
    ax.invert_yaxis()
def paper_bs_art(exp, pos, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    _, axs = plt.subplots(nrows=len(pos)+1, ncols=2, figsize=figsize)
    # dataset
    Bs = exp.resul['objs'][0]['dataset'].Bs
    sns.heatmap(Bs[0], mask=abs(Bs[0]) < ZERO_PRECISION, ax=axs[0, 0],
                cmap='gray', cbar=False, center=0, vmin=0, vmax=4, annot=True)
    sns.heatmap(Bs[1], mask=abs(Bs[1]) < ZERO_PRECISION, ax=axs[0, 1],
                cmap='gray', cbar=False, center=0, vmin=0, vmax=4, annot=True)
    axs[0, 0].tick_params(labelbottom=False, labelleft=False)
    axs[0, 0].set_title('B of group 1')
    axs[0, 1].tick_params(labelbottom=False, labelleft=False)
    axs[0, 1].set_title('B of group 2')
    for _, spine in axs[0, 0].spines.items():
        spine.set_visible(True)
    for _, spine in axs[0, 1].spines.items():
        spine.set_visible(True)
    #gamtl
    for ind, val in enumerate(pos):
        Bs = exp.resul['objs'][val]['GroupAMTL']['resul'][0]['Bs']
        sns.heatmap(Bs[0], mask=abs(Bs[0]) < ZERO_PRECISION, ax=axs[ind+1, 0],
                    cmap='gray', cbar=False, center=0, vmin=0, vmax=4, annot=True)
        sns.heatmap(Bs[1], mask=abs(Bs[1]) < ZERO_PRECISION, ax=axs[ind+1, 1],
                    cmap='gray', cbar=False, center=0, vmin=0, vmax=4, annot=True)
        axs[ind+1, 0].set_title('Estimated B')
        axs[ind+1, 0].tick_params(labelbottom=False, labelleft=False)
        axs[ind+1, 1].set_title('Estimated B')
        axs[ind+1, 1].tick_params(labelbottom=False, labelleft=False)
        for _, spine in axs[ind+1, 0].spines.items():
            spine.set_visible(True)
        for _, spine in axs[ind+1, 1].spines.items():
            spine.set_visible(True)
    axs[len(pos), 0].set_ylabel('from task')
    axs[len(pos), 0].set_xlabel('to task')
    plt.tight_layout()
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/art_bs.pdf')
    plt.show()

def paper_nmse_art(exp, methods, labels, figsize=None, y_log=False):
    """ (Art) Lineplot da metrica no conjunto which_set. """
    df = exp.resul['metrics']
    df = df[df['method'].isin(methods)]
    # plota tudo primeiro
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=df[df.metric == 'nmse'],
                      color='gray',
                      x='dataset_param',
                      y='te',
                      style='method',
                      style_order=methods)
                      #markers=['+', 'x', 'o', 's', 'p', 'D'])

    plt.ylabel('NMSE')
    if y_log:
        ax.set_yscale('log')
        plt.ylabel('log. of NMSE')
    plt.xlabel('Number of Samples')
    plt.legend(labels, loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/art_nmse.pdf')
    plt.show()

def paper_ws(exp, pos, methods=None, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    if not methods:
        methods = [key for key in exp.resul['objs'][pos] if key not in ('dataset', 'hp')]
    ncols = len(methods) + 1
    fig, axs = plt.subplots(nrows=1, ncols=ncols,
                            figsize=figsize)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    w = exp.resul['objs'][pos]['dataset'].W
    mask = abs(w) < ZERO_PRECISION
    sns.heatmap(w, mask=mask, ax=axs[0], cbar=True, cbar_ax=cbar_ax, center=0,
                cmap=cmap_div)
    axs[0].set_title('Original')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    for ind, method in enumerate(methods):
        w = exp.resul['objs'][pos][method]['resul'][0]['W']
        mask = abs(w) < ZERO_PRECISION
        if w is not None:
            sns.heatmap(w, mask=mask, ax=axs[ind+1], cbar=True, cbar_ax=cbar_ax,
                        center=0, cmap=cmap_div)
        axs[ind+1].set_title('{}'.format(method))
        axs[ind+1].set_yticklabels([])
        axs[ind+1].set_xticklabels([])
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_ws.pdf')
    plt.show()

def paper_ws_adni(exp, methods=None, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    pos = 0
    if not methods:
        methods = [key for key in exp.resul['objs'][pos] if key not in ('dataset', 'hp')]
    ncols = len(methods)
    fig, axs = plt.subplots(nrows=1, ncols=ncols,
                            figsize=figsize)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for ind, method in enumerate(methods):
        w = exp.resul['objs'][pos][method]['resul'][0]['W']
        mask = abs(w) < ZERO_PRECISION
        if w is not None:
            sns.heatmap(w, mask=mask, ax=axs[ind], cbar=True, cbar_ax=cbar_ax,
                        center=0, cmap=cmap_div)
        axs[ind].set_title('{}'.format(method))
        axs[ind].set_yticklabels([])
        axs[ind].set_xticklabels([])
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_ws.pdf')
    plt.show()

def paper_structs_adni(exp, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    pos = 0
    methods = [key for key in exp.resul['objs'][pos] if key not in ('dataset', 'hp')]
    methods = ['MyAMTL', 'GroupMTL', 'MSSL', 'MTRL']
    #ncols = np.ceil((len(methods) + 1) / 2).astype('int')
    #fig, axs = plt.subplots(nrows=2, ncols=ncols,
    #        figsize=figsize)
    plt.figure(figsize=figsize)
    for method in methods:
        keys = [key for key in exp.resul['objs'][pos][method]['resul'][0].keys()
                if key != 'W']
        for key in keys:
            w = exp.resul['objs'][pos][method]['resul'][0][key]
            mask = abs(w) < ZERO_PRECISION
            if w is not None:
                ax = sns.heatmap(w, mask=mask, center=0, cmap=cmap_div)
                #sns.heatmap(w, mask=mask, ax=axs[ind+1], cmap=cmap_div, center=0)
                ax.set_title('{} - {}'.format(method, key))
                plt.show()
    #plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_ws.pdf')
    plt.show()

def paper_structs_adni_omegas(exp, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    pos = 0
    methods = [key for key in exp.resul['objs'][pos] if key not in ('dataset', 'hp')]
    #methods = ['MyAMTL', 'MSSL', 'MTRL']
    methods = ['MSSL', 'MTRL']
    ncols = 2
    _, axs = plt.subplots(nrows=1, ncols=ncols,
                          figsize=figsize)
    for ind, method in enumerate(methods):
        keys = [key for key in exp.resul['objs'][pos][method]['resul'][0].keys()
                if key not in ('W', 'C')]
        for key in keys:
            w = exp.resul['objs'][pos][method]['resul'][0][key]
            mask = abs(w) < ZERO_PRECISION
            ax = sns.heatmap(w, mask=mask, ax=axs.flat[ind], cmap=cmap_div,
                             center=0, annot=True)
            ax.set_title('{}'.format(method, key))
            ax.tick_params(labelbottom=False, labelleft=False)
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_omegas.pdf')
    plt.show()

def paper_structs_adni_go(exp, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    pos = 0
    plt.figure(figsize=figsize)
    S = exp.resul['objs'][pos]['GOMTL']['resul'][0]['S']
    mask = abs(S) < ZERO_PRECISION
    #ax = sns.heatmap(S, mask=mask, cmap='gray', center=0, cbar=False)
    ax = sns.heatmap(S, mask=mask, cmap='gray', annot=True, fmt='.2f', cbar=False)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_go.pdf')
    plt.show()

def paper_structs_adni_prec_m(exp, method='MTRL', figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    pos = 0
    plt.figure(figsize=figsize)
    O = exp.resul['objs'][pos][method]['resul'][0]['Omega']
    mask = abs(O) < ZERO_PRECISION
    ax = sns.heatmap(O, mask=mask, cmap='gray', annot=True, fmt='.2f', cbar=False)
    labels = ['TOTAL', 'T30', 'RECOG', 'MMSE', 'ADAS']
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(labelbottom=True, labelleft=True, labelrotation=.45)
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_{}.pdf'.format(method.lower()))
    plt.show()

def paper_structs_adni_amtl(exp, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    pos = 0
    plt.figure(figsize=figsize)
    B = exp.resul['objs'][pos]['MyAMTL']['resul'][0]['B']
    mask = abs(B) < ZERO_PRECISION
    sns.heatmap(B, mask=mask, cmap=cmap_div, center=0, cbar=False)
    #ax.tick_params(labelbottom=False, labelleft=False)
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_amtl.pdf')
    plt.show()

def paper_structs_adni_groupmtl(exp, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    pos = 0
    plt.figure(figsize=figsize)
    Q = exp.resul['objs'][pos]['GroupMTL']['resul'][0]['Q']
    mask = abs(Q) < ZERO_PRECISION
    sns.heatmap(Q, mask=mask, cmap='gray', annot=True)
    #ax.tick_params(labelbottom=False, labelleft=False)
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_groupmtl.pdf')
    plt.show()

def paper_structs_adni_latent(exp, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    pos = 0
    #methods = [key for key in exp.resul['objs'][pos] if key not in ('dataset', 'hp')]
    #methods = ['MyAMTL', 'MSSL', 'MTRL']
    methods = ['GroupMTL', 'GOMTL']
    ncols = 3
    _, axs = plt.subplots(nrows=1, ncols=ncols,
                          figsize=figsize)
    ind = 0
    for method in methods:
        keys = [key for key in exp.resul['objs'][pos][method]['resul'][0].keys()
                if key not in ('W', 'Omega', 'C')]
        for key in keys:
            w = exp.resul['objs'][pos][method]['resul'][0][key]
            mask = abs(w) < ZERO_PRECISION
            ax = sns.heatmap(w, mask=mask, ax=axs.flat[ind], cmap=cmap_div, center=0)
            ind += 1
            ax.set_title('{} - {}'.format(method, key))
    plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_latent.pdf')
    plt.show()

def dataset_description(exp):
    from sklearn.metrics import accuracy_score as acc
    dataset = exp.resul['objs'][0]['dataset']
    display(Markdown('# {} Description#'.format(dataset.name)))
    print('T: {} N: {}'.format(dataset.T, dataset.n))
    print('Amostras por tarefa:')
    df = pd.DataFrame()
    if np.array_equal(dataset.y[0], dataset.y[0].astype(bool)):
        print('Acc quando prediz maior classe')
        for t, yt in enumerate(dataset.y):
            t0 = sum(dataset.y[t] == 0)
            t1 = sum(dataset.y[t] == 1)
            if t0 > t1:
                pred = np.zeros(dataset.y[t].shape[0])
            else:
                pred = np.ones(dataset.y[t].shape[0])
            print('Task {} - {}'.format(t, acc(yt, pred)))
        for t in range(len(dataset.X)):
            s = pd.Series(data=[t,
                                dataset.X[t].shape[0],
                                sum(dataset.y[t] == 1),
                                sum(dataset.y[t] == 0)],
                          index=['Task', 'm', 'class 1', 'class 0'], dtype=int)
            df = df.append(s, ignore_index=True)
        sns.barplot(data=df, x='Task', y='class 1', color='r')
        plt.tight_layout()
        plt.show()
        sns.barplot(data=df, x='Task', y='class 0', color='r')
        plt.tight_layout()
        plt.show()
    else:
        for t, yt in enumerate(dataset.y):
            s = pd.Series(data=[t,
                                dataset.y[t].shape[0]],
                          index=['Task', 'm'], dtype=int)
            df = df.append(s, ignore_index=True)
        sns.barplot(data=df, x='Task', y='m', color='r')
        plt.tight_layout()
        plt.show()
    display(df)

def dataset_description_exp(exp):
    from sklearn.metrics import accuracy_score as acc
    dataset = exp.resul['dataset']
    display(Markdown('# {} Description#'.format(dataset.name)))
    print('T: {} N: {}'.format(dataset.T, dataset.n))
    print('Amostras por tarefa:')
    df = pd.DataFrame()
    if np.array_equal(dataset.y[0], dataset.y[0].astype(bool)):
        print('Acc quando prediz maior classe')
        for t, yt in enumerate(dataset.y):
            t0 = sum(dataset.y[t] == 0)
            t1 = sum(dataset.y[t] == 1)
            if t0 > t1:
                pred = np.zeros(dataset.y[t].shape[0])
            else:
                pred = np.ones(dataset.y[t].shape[0])
            print('Task {} - {}'.format(t, acc(yt, pred)))
        for t in range(len(dataset.X)):
            s = pd.Series(data=[t,
                                dataset.X[t].shape[0],
                                sum(dataset.y[t] == 1),
                                sum(dataset.y[t] == 0)],
                          index=['Task', 'm', 'class 1', 'class 0'], dtype=int)
            df = df.append(s, ignore_index=True)
        sns.barplot(data=df, x='Task', y='class 1', color='r')
        plt.tight_layout()
        plt.show()
        sns.barplot(data=df, x='Task', y='class 0', color='r')
        plt.tight_layout()
        plt.show()
    else:
        for t, yt in enumerate(dataset.y):
            s = pd.Series(data=[t,
                                dataset.y[t].shape[0]],
                          index=['Task', 'm'], dtype=int)
            df = df.append(s, ignore_index=True)
        sns.barplot(data=df, x='Task', y='m', color='r')
        plt.tight_layout()
        plt.show()
    display(df)

def dataset_description_kfold(exp):
    from sklearn.metrics import accuracy_score as acc
    dataset = exp.resul['dataset']
    display(Markdown('# {} Description#'.format(dataset.name)))
    print('T: {} N: {}'.format(dataset.T, dataset.n))
    print('Amostras por tarefa:')
    df = pd.DataFrame()
    if np.array_equal(dataset.y[0], dataset.y[0].astype(bool)):
        print('Acc quando prediz maior classe')
        for t, yt in enumerate(dataset.y):
            t0 = sum(dataset.y[t] == 0)
            t1 = sum(dataset.y[t] == 1)
            if t0 > t1:
                pred = np.zeros(dataset.y[t].shape[0])
            else:
                pred = np.ones(dataset.y[t].shape[0])
            print('Task {} - {}'.format(t, acc(yt, pred)))
        for t in range(len(dataset.T)):
            s = pd.Series(data=[t,
                                dataset.X[t].shape[0],
                                sum(dataset.y[t] == 1),
                                sum(dataset.y[t] == 0)],
                          index=['Task', 'm', 'class 1', 'class 0'], dtype=int)
            df = df.append(s, ignore_index=True)
        sns.barplot(data=df, x='Task', y='class 1', color='r')
        plt.tight_layout()
        plt.show()
        sns.barplot(data=df, x='Task', y='class 0', color='r')
        plt.tight_layout()
        plt.show()
    else:
        for t, yt in enumerate(dataset.y):
            s = pd.Series(data=[t,
                                dataset.y[t].shape[0]],
                          index=['Task', 'm'], dtype=int)
            df = df.append(s, ignore_index=True)
        sns.barplot(data=df, x='Task', y='m', color='r')
        plt.tight_layout()
        plt.show()
        for t, yt in enumerate(dataset.y):
            sns.distplot(yt, color="m")
            plt.title('task {} Y distribution'.format(t))
            plt.show()
    display(df)

def dataset_characteristics(exp, density=True, scatter=True, max_t=np.Inf,
                            dataset=None):
    if not dataset:
        dataset = exp.resul['objs'][0]['dataset']
    display(Markdown('# {} Characteristics #'.format(dataset.name)))
    if density:
        display(Markdown('## Density  of each feature per task ##'.format(dataset.name)))
        for t, Xt in enumerate(dataset.X):
            if t > max_t:
                break
            display(Markdown('### task {} ##'.format(t)))
            df = pd.DataFrame(Xt)
            df.plot(kind='density', subplots=True, sharex=True)
            plt.show()
    if scatter:
        display(Markdown('## Scatter per task ##'.format(dataset.name)))
        for t, Xt in enumerate(dataset.X):
            if t > max_t:
                break
            display(Markdown('### task {} ##'.format(t)))
            df = pd.DataFrame(Xt)
            scatter_matrix(df, figsize=(15, 15))
            plt.show()

def dataset_characteristics_exp(exp, density=True, scatter=True, max_t=np.Inf,
                                dataset=None):
    if not dataset:
        dataset = exp.resul['dataset']
    display(Markdown('# {} Characteristics #'.format(dataset.name)))
    if density:
        display(Markdown('## Density  of each feature per task ##'.format(dataset.name)))
        for t, Xt in enumerate(dataset.X):
            if t > max_t:
                break
            display(Markdown('### task {} ##'.format(t)))
            df = pd.DataFrame(Xt)
            df.plot(kind='density', subplots=True, sharex=True)
            plt.show()
    if scatter:
        display(Markdown('## Scatter per task ##'.format(dataset.name)))
        for t, Xt in enumerate(dataset.X):
            if t > max_t:
                break
            display(Markdown('### task {} ##'.format(t)))
            df = pd.DataFrame(Xt)
            scatter_matrix(df, figsize=(15, 15))
            plt.show()

def dataset_characteristics_kfold(exp, density=True, scatter=True, max_t=np.Inf):
    dataset = exp.resul['dataset']
    dataset_characteristics(exp, density, scatter, max_t, dataset)

def dataset_var_features(exp):
    data = exp.resul['objs'][0]['dataset']
    sns.boxplot(data=data.X[0])
    plt.title('boxplot cada feature')
    plt.show()

def dataset_var_features_exp(exp):
    data = exp.resul['dataset']
    sns.boxplot(data=data.X[0])
    plt.title('boxplot cada feature')
    plt.show()

def dataset_var_features_kfold(exp):
    data = exp.resul['dataset']
    sns.boxplot(data=data.X[0])
    plt.title('boxplot cada feature')
    plt.show()

def print_cross_val(exp):
    local_methods = [key for key in exp.resul['objs'][0] \
                        if key not in ('dataset', 'hp', 'metrics', 'task_metrics')]
    for method in local_methods:
        print('{}'.format(method))
        print('Best params')
        display(exp.resul['objs'][0][method]['best_params'])
        display(exp.resul['objs'][0][method]['hyper_params'])

def plot_cross_val(exp, pos=0):
    local_methods = [key for key in exp.resul['objs'][pos] \
                        if key not in ('dataset', 'hp', 'metrics', 'task_metrics')]
    for method in local_methods:
        print('{}'.format(method))
        print('Best params')
        df = exp.resul['objs'][pos][method]['hyper_params']
        params = [col for col in df.columns.values if col not in ('metric', 'val', 'tr')]
        if len(params) == 1:
            plt.figure()
            sns.lineplot(x=params[0], y='val', data=df, palette='Paired',
                         markers=True)
            plt.show()
        elif len(params) == 2:
            plt.figure()
            sns.lineplot(x=params[0], y='val', hue=params[1], data=df,
                         palette='Paired', markers=True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()
        elif len(params) == 3:
            for param_val in df[params[2]].unique():
                print('Quando {} == {}'.format(params[2], param_val))
                plt.figure()
                sns.lineplot(x=params[0], y='val', hue=params[1],
                             data=df.loc[df[params[2]] == param_val],
                             palette='Paired', markers=True)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.show()

def plot_cross_val_exp(exp):
    local_methods = [key for key in exp.resul.keys()\
                        if key not in ('dataset', 'hp', 'metrics', 'task_metrics')]
    for method in local_methods:
        print('{}'.format(method))
        print('Best params')
        df = exp.resul[method]['hyper_params']
        params = [col for col in df.columns.values if col not in ('metric', 'val', 'tr')]
        if len(params) == 1:
            plt.figure()
            sns.lineplot(x=params[0], y='val', data=df, palette='Paired',
                         markers=True)
            plt.show()
        elif len(params) == 2:
            plt.figure()
            sns.lineplot(x=params[0], y='val', hue=params[1], data=df,
                         palette='Paired', markers=True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()
        elif len(params) == 3:
            for param_val in df[params[2]].unique():
                print('Quando {} == {}'.format(params[2], param_val))
                plt.figure()
                sns.lineplot(x=params[0], y='val', hue=params[1],
                             data=df.loc[df[params[2]] == param_val],
                             palette='Paired', markers=True)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.show()

def plot_cross_go(exp, pos=0):
    method = 'GOMTL'
    print('Best params')
    df = exp.resul['objs'][pos][method]['hyper_params']
    for param_val in df['nb_latvars'].unique():
        print('Quando nb_latvars == {}'.format(param_val))
        plt.figure()
        sns.lineplot(x='rho_1', y='val', hue='rho_2',
                     data=df.loc[df['nb_latvars'] == param_val],
                     markers=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

def plot_cross_go_exp(exp):
    method = 'GOMTL'
    print('Best params')
    df = exp.resul[method]['hyper_params']
    for param_val in df['nb_latvars'].unique():
        print('Quando nb_latvars == {}'.format(param_val))
        plt.figure()
        sns.lineplot(x='rho_1', y='val', hue='rho_2',
                     data=df.loc[df['nb_latvars'] == param_val],
                     markers=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

def plot_cross_amtl(exp, pos=0, remove_lamb=None):
    method = 'MyAMTL'
    print('Best params')
    df = exp.resul['objs'][pos][method]['hyper_params']
    params = [col for col in df.columns.values if col not in ('metric', 'val', 'tr')]
    for val in remove_lamb:
        df = df[df['lamb'] != val]
    plt.figure()
    sns.lineplot(x=params[0], y='val', hue=params[1], data=df,
                 palette='Paired', markers=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot_cross_amtl_exp(exp, remove_lamb=None):
    method = 'MyAMTL'
    print('Best params')
    df = exp.resul[method]['hyper_params']
    params = [col for col in df.columns.values if col not in ('metric', 'val', 'tr')]
    for val in remove_lamb:
        df = df[df['lamb'] != val]
    plt.figure()
    sns.lineplot(x=params[0], y='val', hue=params[1], data=df,
                 palette='Paired', markers=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def box_metric(exp, metric='macc', which_set='te', y_log=False, figsize=None):
    """ (Art) Boxplot da metrica no conjunto which_set. """
    df = exp.resul['metrics']
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df[df.metric == metric], hue='method', x='dataset_param',
                     y=which_set)
    if y_log:
        ax.set_yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()

def box_metric_kfold(exp, metric='macc', which_set='te', y_log=False, figsize=None):
    """ (Art) Boxplot da metrica no conjunto which_set. """
    df = exp.resul['metrics']
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df[df.metric == metric], hue='method', x='fold',
                     y=which_set)
    if y_log:
        ax.set_yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()

def line_metric(exp, figsize=None, metric='macc', which_set='te', divide=True,
                each=False, y_log=False):
    """ (Art) Lineplot da metrica no conjunto which_set. """
    if each:
        print('not implemented yet')
    else:
        df = exp.resul['metrics']
        # plota tudo primeiro
        plt.figure(figsize=figsize)
        ax = sns.lineplot(data=df[df.metric == metric],
                          hue='method',
                          x='dataset_param',
                          y=which_set,
                          palette='deep')
        if y_log:
            ax.set_yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        if divide:
            import math
            methods = df.method.unique()
            for i in np.arange(math.ceil(len(methods)/5)):
                start = i * 5
                stop = start + 5
                plt.figure(figsize=figsize)
                ax = sns.lineplot(data=df[df.metric == metric][df['method'].isin(methods[start:stop])],
                                  hue='method',
                                  style='method',
                                  x='dataset_param',
                                  y=which_set,
                                  palette='deep')
                if y_log:
                    ax.set_yscale('log')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.show()

def line_metric_specific(exp, methods, figsize=None, metric='macc', which_set='te', divide=True,
                         y_log=False):
    """ (Art) Lineplot da metrica no conjunto which_set. """
    df = exp.resul['metrics']
    df = df[df['method'].isin(methods)]
    # plota tudo primeiro
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=df[df.metric == metric],
                      hue='method',
                      x='dataset_param',
                      y=which_set,
                      palette='deep')
    if y_log:
        ax.set_yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    if divide:
        import math
        methods = df.method.unique()
        for i in np.arange(math.ceil(len(methods)/5)):
            start = i * 5
            stop = start + 5
            plt.figure(figsize=figsize)
            ax = sns.lineplot(data=df[df.metric == metric][df['method'].isin(methods[start:stop])],
                              hue='method',
                              style='method',
                              x='dataset_param',
                              y=which_set,
                              palette='deep')
            if y_log:
                ax.set_yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()

def table(exp, param_val):
    df = exp.resul['metrics']
    df = df[df.dataset_param == param_val]
    df_g = df.groupby(['dataset_param', 'method'])
    display(df_g.agg([np.mean, np.std]))

def table_exp(exp):
    df = exp.resul['metrics']
    df_g = df.groupby(['method'])
    display(df_g.agg([np.mean, np.std]))

def table_kfold(exp, metric='nmse'):
    df = exp.resul['metrics'][exp.resul['metrics']['metric'] == metric]
    df_g = df.groupby(['metric', 'method'])
    return df_g['te', 'tr'].agg([np.mean, np.std])

def box_metric_real(exp, metric='macc', which_set='te', y_log=False):
    """ (Real) Boxplot da metrica no conjunto which_set. """
    df = exp.resul['metrics']
    ax = sns.boxplot(data=df[df.metric == metric], x='method', y=which_set)
    if y_log:
        ax.set_yscale('log')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()

def box_metric_per_val(exp, param_val=np.Inf, metric='macc', which_set='te',
                       figsize=None, y_log=False):
    """ (Art) Boxplot da metrica no conjunto which_set variando dataset_param. """
    df = exp.resul['metrics']
    df2 = df[df.metric == metric]
    df3 = df2[df2.dataset_param == param_val]
    df_g = df3.groupby('method').agg(np.mean)
    inds = df_g.sort_values('te').index
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df3, x='method', y=which_set, order=inds)
    if y_log:
        ax.set_yscale('log')
    plt.setp(ax.get_xticklabels(), rotation=45)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def box_metric_task(exp, param_val=np.Inf, metric='acc', which_set='te',
                    y_log=False):
    """ (Art) Boxplot da metrica por tarefa no conjunto which_set no param_val. """
    df = exp.resul['task_metrics']
    df2 = df[df.metric == metric]
    df3 = df2[df2.dataset_param == param_val]
    ax = sns.boxplot(data=df3, x='task', y=which_set, hue='method')
    if y_log:
        ax.set_yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def box_metric_task_real(exp, metric='acc', which_set='te', y_log=False):
    """ (Real) Boxplot da metrica por tarefa no conjunto which_set. """
    df = exp.resul['task_metrics']
    df2 = df[df.metric == metric]
    ax = sns.boxplot(data=df2, x='task', y=which_set, hue='method')
    if y_log:
        ax.set_yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def bar_metric_task(exp, param_val=np.Inf, metric='acc', which_set='te',
                    figsize=None, y_log=False):
    """ (Art) Barplot da metrica por tarefa no conjunto which_set no param_val. """
    df = exp.resul['task_metrics']
    df2 = df[df.metric == metric]
    df3 = df2[df2.dataset_param == param_val]
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df3, x='task', y=which_set, hue='method')
    if y_log:
        ax.set_yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def bar_metric_task_real(exp, metric='acc', which_set='te', figsize=None,
                         y_log=False):
    """ (Real) Barplot da metrica por tarefa no conjunto which_set. """
    df = exp.resul['task_metrics']
    df2 = df[df.metric == metric]
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df2, x='task', y=which_set, hue='method')
    if y_log:
        ax.set_yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def table_metric_task_real(exp, metric='mse'):
    """ (Real) Tabela da metrica por tarefa no conjunto which_set. """
    df = exp.resul['task_metrics']
    df = df.drop(['run', 'fold'], axis=1)
    grouped_df = df[df['metric'] == metric].groupby(['task', 'method'])
    return grouped_df.agg([np.mean, np.std])

def heat_ws_real(exp, figsize=None):
    """ (Real) Heatmap dos Ws de todos os métodos do experimento."""
    n_cols = len(exp.strategies.get_list())
    fig, axs = plt.subplots(1, n_cols, figsize=figsize)
    local_methods = [key for key in exp.resul if key not in ('dataset', 'hp',
                                                             'metrics', 'task_metrics')]
    for ind, method in enumerate(local_methods):
        w = exp.resul[method]['resul'][0]['W']
        mask = abs(w) < 10e-5
        sns.heatmap(w, mask=mask, ax=axs[ind], cmap=cmap_div, center=0)
        axs[ind].set_title('{}'.format(method))
    fig.tight_layout()
    plt.show()

def heat_ws_real_kfold(exp, figsize=None):
    """ (Real) Heatmap dos Ws de todos os métodos do experimento."""
    n_cols = len(exp.strategies.get_list())
    fig, axs = plt.subplots(1, n_cols, figsize=figsize)
    local_methods = [key for key in exp.resul['objs'][0] \
                        if key not in ('hp', 'metrics', 'task_metrics')]
    for ind, method in enumerate(local_methods):
        w = exp.resul['objs'][0][method]['resul'][0]['W']
        mask = abs(w) < 10e-5
        sns.heatmap(w, mask=mask, ax=axs[ind], cmap=cmap_div, center=0)
        axs[ind].set_title('{}'.format(method))
    fig.tight_layout()
    plt.show()

def heat_ws(exp, pos=0, figsize=None):
    """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
    if hasattr(exp.resul['objs'][pos]['dataset'], 'W'):
        w = exp.resul['objs'][pos]['dataset'].W
        mask = abs(w) < ZERO_PRECISION
        plt.figure(figsize=figsize)
        sns.heatmap(w, mask=mask, cmap=cmap_div, center=0)
        plt.title('Original')
        plt.show()
    local_methods = [key for key in exp.resul['objs'][pos] if key not in ('dataset', 'hp')]
    for method in local_methods:
        w = exp.resul['objs'][pos][method]['resul'][0]['W']
        mask = abs(w) < ZERO_PRECISION
        plt.figure(figsize=figsize)
        sns.heatmap(w, mask=mask, cmap=cmap_div, center=0)
        plt.title('{}'.format(method))
        plt.show()

def heat_bs_real(exp, figsize=None):
    """ (Real) Heatmap dos Bs de todos os métodos GroupAMTL do experimento."""
    local_methods = [key for key in exp.resul if key not in ('dataset', 'hp')]
    method_names = [method for method in local_methods if 'GroupAMTL' in method]
    for method in method_names:
        Bs = exp.resul[method]['resul'][0]['Bs']
        n_cols = len(Bs)
        fig, axs = plt.subplots(1, n_cols, figsize=figsize)
        for ind_g, B in enumerate(Bs):
            b_est = np.abs(B) < 10e-5
            if n_cols == 1:
                ax_ref = axs
            else:
                ax_ref = axs[ind_g]
            sns.heatmap(B, mask=b_est, ax=ax_ref, cmap=cmap, center=0)
            ax_ref.set_ylabel('from task')
            ax_ref.set_xlabel('to task')
            ax_ref.set_title('Bg{}'.format(ind_g))
        fig.suptitle(method)
        fig.tight_layout()
        plt.show()

def heat_bs_real_kfold(exp, figsize=None):
    """ (Real) Heatmap dos Bs de todos os métodos GroupAMTL do experimento."""
    local_methods = [key for key in exp.resul['objs'][0] if key not in 'hp']
    method_names = [method for method in local_methods if 'GroupAMTL' in method]
    for method in method_names:
        Bs = exp.resul['objs'][0][method]['resul'][0]['Bs']
        n_cols = len(Bs)
        fig, axs = plt.subplots(1, n_cols, figsize=figsize)
        for ind_g, B in enumerate(Bs):
            b_est = np.abs(B) < 10e-5
            if n_cols == 1:
                ax_ref = axs
            else:
                ax_ref = axs[ind_g]
            sns.heatmap(B, mask=b_est, ax=ax_ref, cmap=cmap_div, center=0)
            ax_ref.set_ylabel('from task')
            ax_ref.set_xlabel('to task')
            ax_ref.set_title('Bg{}'.format(ind_g))
        fig.suptitle(method)
        fig.tight_layout()
        plt.show()

def heat_bs(exp, pos=0, figsize=None, show_ref=True):
    """ (Art) Heatmap dos Bs de todos os métodos GroupAMTL do experimento."""
    local_methods = [key for key in exp.resul['objs'][pos] if key not in ('dataset', 'hp')]
    method_names = [method for method in local_methods if 'GroupAMTL' in method]
    # Bs dataset
    if show_ref and hasattr(exp.resul['objs'][pos]['dataset'], 'Bs'):
        Bs = exp.resul['objs'][pos]['dataset'].Bs
        n_cols = len(Bs)
        fig, axs = plt.subplots(1, n_cols, figsize=figsize)
        for ind_g, B in enumerate(Bs):
            Aa = (B + B.T)/2
            As = (B - B.T)/2
            sim = np.linalg.norm(As) / np.linalg.norm(Aa)
            b_est = np.abs(B) < 10e-5
            if n_cols == 1:
                ax_ref = axs
            else:
                ax_ref = axs[ind_g]
            sns.heatmap(B, mask=b_est, ax=ax_ref, cmap=cmap, center=0)
            ax_ref.set_ylabel('from task')
            ax_ref.set_xlabel('to task')
            ax_ref.set_title('Bg{} sim:{:.4f}'.format(ind_g, sim))
        fig.suptitle('Generated Bs')
        fig.tight_layout()
        plt.show()
    for method in method_names:
        Bs = exp.resul['objs'][pos][method]['resul'][0]['Bs']
        n_cols = len(Bs)
        fig, axs = plt.subplots(1, n_cols, figsize=figsize)
        for ind_g, B in enumerate(Bs):
            Aa = (B + B.T)/2
            As = (B - B.T)/2
            sim = np.linalg.norm(As) / np.linalg.norm(Aa)
            b_est = np.abs(B) < 10e-5
            if n_cols == 1:
                ax_ref = axs
            else:
                ax_ref = axs[ind_g]
            sns.heatmap(B, mask=b_est, ax=ax_ref, cmap=cmap_div, center=0)
            ax_ref.set_ylabel('from task')
            ax_ref.set_xlabel('to task')
            ax_ref.set_title('Bg{} sim: {:.4f}'.format(ind_g, sim))
        fig.suptitle(method)
        fig.tight_layout()
        plt.show()

def print_costs(exp, pos=0):
    """
        Imprime os custos da formulação do GroupAMTL, utilizando Ws e Bs estimados
        em contraste com os Ws e Bs utilizados na geração do dataset.

        Args:
            exp (codes.Experiment): experimento a ser analisado.
            pos (int): posiçao do vetor resul a ser exibida.
    """
    dataset = exp.resul['objs'][pos]['dataset']
    params = exp.resul['objs'][pos]['GroupAMTL']['best_params']
    print('Utilizando params de GroupAMTL na run 0')
    display(params)
    X = dataset.get_test()['X']
    y = dataset.get_test()['y']
    gamtl_ref = GroupAMTL(dataset.groups)
    gamtl_ref.W = dataset.W
    gamtl_ref.Bs = np.array(dataset.Bs)
    gamtl_ref.set_params(**params)
    costs = pd.Series()
    costs['Dataset'] = gamtl_ref._cost_function(X, y)
    for method, _ in exp.methods:
        if 'GroupAMTL' not in method.__name__:
            continue
        gamtl = method(groups=dataset.groups)
        gamtl.W = exp.resul['objs'][pos][gamtl.name]['resul'][0]['W']
        gamtl.Bs = exp.resul['objs'][pos][gamtl.name]['resul'][0]['Bs']
        gamtl.set_params(**params)
        costs[gamtl.name] = gamtl._cost_function(X, y)
    for name, val in costs.items():
        print('{0:50s} {1:.8f}'.format(name, val))

def box_times(exp, pos=0, figsize=None):
    """ (Art) Boxplot com o tempo de execução dos métodos do exp na posição pos. """
    df = pd.DataFrame()
    local_methods = [key for key in exp.resul['objs'][pos] if key not in ('dataset', 'hp')]
    for meth_name in local_methods:
        df[meth_name] = pd.Series(exp.resul['objs'][pos][meth_name]['time'],
                                  name=meth_name)
    df = df.reindex_axis(df.mean().sort_values().index, axis=1)
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel('time in seconds')
    plt.show()

def box_times_exp(exp, figsize=None):
    """ (Art) Boxplot com o tempo de execução dos métodos do exp na posição pos. """
    df = pd.DataFrame()
    local_methods = [key for key in exp.resul.keys() \
                        if key not in ('dataset', 'metrics', 'task_metrics', 'hp')]
    print(local_methods)
    for meth_name in local_methods:
        df[meth_name] = pd.Series(exp.resul[meth_name]['time'],
                                  name=meth_name)
    df = df.reindex_axis(df.mean().sort_values().index, axis=1)
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel('time in seconds')
    plt.show()

def box_times_real(exp, pos=0):
    """ (Art) Boxplot com o tempo de execução dos métodos do exp na posição pos. """
    df = pd.DataFrame()
    for method, _ in exp.methods:
        meth = method()
        df[meth.name] = pd.Series(exp.resul['objs'][pos][meth.name]['time'],
                                  name=meth.name)
    ax = sns.boxplot(data=df)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel('time in seconds')
    plt.show()

def imprime_custos_real(exp, pos=0):
    """
        Imprime os custos da formulação do GroupAMTL, utilizando Ws e Bs estimados
        em contraste com os Ws e Bs utilizados na geração do dataset.
        Args:
            exp (codes.Experiment): experimento a ser analisado.
            pos (int): posiçao do vetor resul a ser exibida.
    """
    params = exp.resul[pos]['GroupAMTL']['best_params']
    gamtl = GroupAMTL(groups=params['groups'])
    gamtl.W = exp.resul[pos]['GroupAMTL']['resul'][0]['W']
    gamtl.Bs = exp.resul[pos]['GroupAMTL']['resul'][0]['Bs']
    gamtl.set_params(**params)
    dataset = exp.resul['objs'][pos]['dataset']
    X = dataset.get_train()['X']
    y = dataset.get_train()['y']
    gamtl_cost = gamtl._cost_function(X, y)
    print('Custo com parametros estimados no test: {}'.format(gamtl_cost))
