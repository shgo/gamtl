"""
Module that contains classes that plot all needed graphics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics import accuracy_score as acc
from codes.experiments import ExperimentMTL, ExperimentKfold, ExperimentVarParam
pd.options.display.float_format = '{:,.3f}'.format
cmap_div = sns.color_palette("RdBu_r", n_colors=30)
cmap = sns.light_palette("purple", reverse=False)
ZERO_PRECISION = 1e-30


class Plotter:
    """
    """
    def __init__(self, exp):
        self.exp = exp

    def dataset_description(self, dataset):
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
                              index=['Task', 'm', 'class 1', 'class 0'],
                              dtype=int)
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

    def dataset_var_feats(self, dataset):
        sns.boxplot(data=dataset.X[0])
        plt.title('boxplot cada feature')
        plt.show()


class PlotterExp(Plotter):
    """
    """
    def __init__(self, exp):
        assert isinstance(exp, ExperimentMTL)
        super().__init__(exp)

    def dataset_description(self):
        super().dataset_description(self.exp.resul['dataset'])

    def dataset_var_feats(self):
        super().dataset_var_feats(self.exp.resul['dataset'])

    def box_times(self, figsize=None):
        """
        (Art) Boxplot com o tempo de execução dos métodos do exp na posição pos
        Args:
            pos (int):
            figsize (float, float): width, height in inches.
        """
        local_methods = [key for key in self.exp.resul.keys()
                         if key not in ('dataset', 'hp',
                                        'task_metrics', 'metrics')]
        df = pd.DataFrame()
        for meth_name in local_methods:
            df[meth_name] = pd.Series(self.exp.resul[meth_name]['time'],
                                      name=meth_name)
        df = df.reindex_axis(df.mean().sort_values().index, axis=1)
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df)
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.ylabel('time in seconds')
        plt.show()

    def plot_cross_val(self, figsize=(9, 3.5)):
        """ Plots cross-validation curves for all methods.
        Just enough to see if the grid was good. """
        local_methods = [key for key in self.exp.resul.keys()
                         if key not in ('dataset', 'hp',
                                        'metrics', 'task_metrics')]
        for method in local_methods:
            print('{}'.format(method.upper()))
            print('Best params')
            df = self.exp.resul[method]['hyper_params']
            params = [col for col in df.columns.values
                      if col not in ('metric', 'val', 'tr')]
            if len(params) == 1:
                dfg = df.groupby(params)
                means = dfg.aggregate('mean')
                stds = dfg.aggregate('std')
                plt.figure()
                plt.plot(means['tr'].index, means['tr'], label='tr')
                plt.fill_between(means['tr'].index.values,
                                 means['tr'] - stds['tr'],
                                 means['tr'] + stds['tr'],
                                 color='gray', alpha=0.2)

                plt.plot(means['val'], label='val')
                plt.fill_between(means['val'].index.values,
                                 means['val'] - stds['val'],
                                 means['val'] + stds['val'],
                                 color='gray', alpha=0.2)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.xlabel('Error')
                plt.xlabel(params[0])
                plt.show()
            elif len(params) == 2:
                for par_2 in df[params[1]].unique():
                    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
                    ldf = df[df[params[1]] == par_2]
                    dfg = df.groupby(params[0])
                    means = dfg.aggregate('mean')
                    stds = dfg.aggregate('std')
                    ax1.plot(means['tr'], label='{}: {}'.format(params[1], par_2))
                    ax1.fill_between(means['tr'].index.values,
                                     means['tr'] - stds['tr'],
                                     means['tr'] + stds['tr'],
                                     color='gray', alpha=0.2)
                    ax2.plot(means['val'], label='{}: {}'.format(params[1], par_2))
                    ax2.fill_between(means['val'].index.values,
                                     means['val'] - stds['val'],
                                     means['val'] + stds['val'],
                                     color='gray', alpha=0.2)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax1.set_title('TR {} = {:.4f}'.format(params[1], par_2))
                    ax2.set_title('VAL {} = {:.4f}'.format(params[1], par_2))
                    ax2.yaxis.set_tick_params(labelleft=True)
                    ax1.set_ylabel('error')
                    ax1.set_xlabel(params[0])
                    plt.show()
            elif len(params) == 3:
                for lamb_3 in df[params[2]].unique():
                    local_df = df[df[params[2]] == lamb_3]
                    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)

                    for lamb_2 in local_df[params[1]].unique():
                        ldf = local_df[local_df[params[1]] == lamb_2]
                        dfg = ldf.groupby(params[0])
                        means = dfg.aggregate('mean')
                        stds = dfg.aggregate('std')

                        ax1.plot(means['tr'],
                                 label='{} = {:.4f}'.format(params[1], lamb_2))
                        ax1.fill_between(means['tr'].index.values,
                                         means['tr'] - stds['tr'],
                                         means['tr'] + stds['tr'],
                                         color='gray', alpha=0.2)

                        ax2.plot(means['val'],
                                 label='{} = {:.4f}'.format(params[1], lamb_2))
                        ax2.fill_between(means['val'].index.values,
                                         means['val'] - stds['val'],
                                         means['val'] + stds['val'],
                                         color='gray', alpha=0.2)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax1.set_title('TR {} = {:.4f}'.format(params[2], lamb_3))
                    ax2.set_title('VAL {} = {:.4f}'.format(params[2], lamb_3))
                    ax2.yaxis.set_tick_params(labelleft=True)
                    ax1.set_ylabel('error')
                    ax1.set_xlabel(params[0])
                    plt.show()

    def table_metric(self):
        """ Prints a table with all overall metrics per method
            (mean, std over all runs) using train and test set. """
        df = self.exp.resul['metrics']
        df_g = df.groupby(['method'])
        display(df_g.agg([np.mean, np.std]))

    def box_metric(self, metric='macc', which_set='te', y_log=False):
        """ (Real) Boxplot da metrica no conjunto which_set. """
        df = self.exp.resul['metrics']
        ax = sns.boxplot(data=df[df.metric == metric], x='method', y=which_set)
        if y_log:
            ax.set_yscale('log')
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.show()

    def bar_metric_task(self, metric='acc', which_set='te', figsize=None,
                        y_log=False, labels=None):
        """ (Real) Barplot da metrica por tarefa no conjunto which_set.

        Args:
            metric (str): metric to be plotted.
            which_set (str): 'te' or 'tr'.
            figsize (float, float): width, height in inches.
            y_log (bool): y axis in logscale.
        """
        df = self.exp.resul['task_metrics']
        df2 = df[df.metric == metric]
        plt.figure(figsize=figsize)
        ax = sns.barplot(data=df2, x='task', y=which_set, hue='method')
        if y_log:
            ax.set_yscale('log')
        if labels:
            ax.set_xticklabels(labels)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def heat_ws(self, figsize=None):
        """
        Heatmap of all parameter matrices W of all methods.

        Args:
            figsize (float, float): width, height in inches.
        """
        n_cols = len(self.exp.strategies.get_list())
        fig, axs = plt.subplots(1, n_cols, figsize=figsize)
        local_methods = [key for key in self.exp.resul
                         if key not in ('dataset', 'hp',
                                        'metrics', 'task_metrics')]
        for ind, method in enumerate(local_methods):
            w = self.exp.resul[method]['resul'][0]['W']
            mask = abs(w) < 10e-5
            sns.heatmap(w, mask=mask, ax=axs[ind], cmap=cmap_div, center=0)
            axs[ind].set_title('{}'.format(method))
        fig.tight_layout()
        plt.show()

    def print_top_groups(self, k, method='GroupAMTL_nr', figsize=None,
                         labels=None, kind='heatmap'):
        """
        Heatmap of the k biggest relationship matrices, in terms of matricial
        norm1 of method.

        Args:
            k (int): top-k norm1 groups.
            method (str): method to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        norms = self._compute_b_norms(method)
        inds = np.argsort(norms)[-1*k:]
        for ind_g in inds:
            if kind == 'heatmap':
                self._print_group(ind_g, [method], figsize=figsize,
                                  labels=labels)
            elif kind == 'graph':
                self._print_group_graph(ind_g, [method], figsize=figsize)

    def print_groups(self, inds, methods, figsize=None, labels=None,
                     kind='heatmap'):
        """
        Heatmap of relationship matrices for all groups in inds.

        Args:
            inds (list of int): list with index of groups to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        for ind_g in inds:
            if kind == 'heatmap':
                self._print_group(ind_g, methods, figsize=figsize,
                                  labels=labels)
            elif kind == 'graph':
                self._print_group_graph(ind_g, methods, figsize=figsize,
                                        labels=labels)

    def _print_group(self, ind_g, methods=None, figsize=None, labels=None):
        """
        INTERNAL METHOD that prints ind_g group of methods.

        Args:
            ind_g (int): index of group to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        if not labels:
            labels = ['TOTAL', 'T30', 'RECOG', 'MMSE', 'ADAS']
        fold = 0
        for ind, method in enumerate(methods):
            B = self.exp.resul[method]['resul'][0]['Bs'][ind_g]
            print('{} fold {}: {}'.format(methods[ind], fold, ind_g))
            fig = plt.figure(figsize=figsize)
            ax = sns.heatmap(B, mask=abs(B) < 1e-4, center=0, cmap='gray_r',
                             annot=True, fmt='.2f', cbar=False)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.tick_params(labelbottom=True, labelleft=True, labelrotation=.45)

            for _, spine in ax.spines.items():
                spine.set_visible(True)
            plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_{}_bg{}.pdf'.format(method.lower(), ind_g))
            plt.show()

    def _compute_b_norms(self, method='GroupAMTL_nr'):
        """
        Computes the norm of all Bs over all fold.

        Args:
            method (str): method name

        Returns:
            norms (np.array): n_folds x n_groups. Each element f,g is
              the norm1 of Bg matrix in fold f.
            vals (np.array): shape (n_groups) with the mean of norm1
              of Bg matrix over all folds.
        """
        Bs = self.exp.resul[method]['resul'][0]['Bs']
        n_groups = len(Bs)
        norms = np.zeros(n_groups)
        for g in range(n_groups):
            norms[g] = sum(sum(abs(Bs[g])))
        return norms

    def _print_group_graph(self, ind_g, methods, figsize=None):
        """ INTERNAL METHOD that prints a graph for Bs[ind_g].

        Args:
            ind_g (int): index of group to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        fold = 0
        for ind, method in enumerate(methods):
            B = self.exp.resul[method]['resul'][0]['Bs'][ind_g]
            print('{} fold {}: {}'.format(methods[ind], fold, ind_g))
            fig = plt.figure(figsize=figsize)
            G = nx.DiGraph()
            for i in range(B.shape[0]):
                G.add_node(i)

            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    G.add_edge(i, j, weight=B[i, j])

            pos=nx.spring_layout(G) # positions for all nodes
            # nodes
            nx.draw_networkx_nodes(G, pos, node_size=700)
            # edges
            nx.draw_networkx_edges(G, pos, width=6)
            nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
            #plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_{}_bg{}.pdf'.format(method.lower(), ind_g))
            plt.axis('off')
            plt.show()

    def paper_hinton_bs(self, methods, max_weight=None, figsize=None,
                        save=False, run=0):
        """
        (Art) Hintonmap dos Ws de todos os métodos do experimento.
        Code from: https://matplotlib.org/gallery/specialty_plots/hinton_demo.html
        """
        Bs = self.exp.resul[methods[0]]['resul'][0]['Bs']
        fig, axs = plt.subplots(nrows=len(Bs), ncols=len(methods),
                                figsize=figsize)

        for i, method in enumerate(methods):
            Bs = self.exp.resul[method]['resul'][0]['Bs']
            for j, B in enumerate(Bs):
                self.hinton(B.T, max_weight=max_weight, ax=axs[j, i])
                axs[j, i].tick_params(
                    axis='both',          # changes apply to both axis
                    which='both',      # both major and minor ticks affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    left=False,         # ticks along the left edge are off
                    right=False,         # ticks along the right edge are off
                    labelbottom=False, # labels along the bottom edge are off
                    labelleft=False) # labels along the bottom edge are off
            axs[0, i].set_title('{}'.format(method))
        plt.tight_layout()
        if save:
            plt.savefig('/home/churros/pasta/codes/gamtl/figs/art_hinton_bs.pdf')
        plt.show()

    def paper_ws(self, methods=None, figsize=None, save=False):
        """
        """
        if not methods:
            methods = [i for i in self.exp.resul.keys()\
                       if i not in ('dataset', 'hp', 'metrics', 'task_metrics')]
        qtd_methods = len(methods)
        fig, axs = plt.subplots(nrows=1, ncols=qtd_methods,
                                figsize=figsize)
        tick_params = {
            'axis': 'both',        # changes apply to both axis
            'which': 'both',       # both major and minor ticks are affected
            'bottom': False,       # ticks along the bottom edge are off
            'top': False,          # ticks along the top edge are off
            'left': False,         # ticks along the left edge are off
            'right': False,        # ticks along the right edge are off
            'labelbottom': False,  # labels along the bottom edge are off
            'labelleft': False     # labels along the bottom edge are off
        }
        for j, method in enumerate(methods):
            w = self.exp.resul[method]['resul'][0]['W']
            mask = abs(w) < ZERO_PRECISION
            sns.heatmap(w, mask=mask, cmap=cmap_div, center=0,
                        ax=axs[j], cbar=False)
            axs[j].tick_params(**tick_params)
            axs[j].set_title('{}'.format(method))
        for ax in axs:
            for _, spine in ax.spines.items():
                spine.set_visible(True)
        if save:
            plt.savefig('/home/churros/pasta/codes/gamtl/figs/art_ws.pdf')
        plt.plot()

    def hinton(self, matrix, max_weight=None, ax=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        ax = ax if ax is not None else plt.gca()
        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('white')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=9))

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if abs(w) < ZERO_PRECISION else 'black'
            size = np.sqrt(np.abs(w) / max_weight) + 0.0001
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)
        ax.autoscale_view()
        ax.invert_yaxis()


class PlotterExpKFold(Plotter):
    """
    """
    def __init__(self, exp):
        assert isinstance(exp, ExperimentMTL)
        super().__init__(exp)

    def dataset_description(self):
        super().dataset_description(self.exp.resul['dataset'])

    def dataset_var_feats(self):
        super().dataset_var_feats(self.exp.resul['dataset'])

    def box_times(self, figsize=None):
        """
        (Art) Boxplot com o tempo de execução dos métodos do exp na posição pos
        Args:
            pos (int):
            figsize (float, float): width, height in inches.
        """
        local_methods = [key for key in self.exp.resul.keys()
                         if key not in ('dataset', 'hp',
                                        'task_metrics', 'metrics')]
        df = pd.DataFrame()
        for meth_name in local_methods:
            df[meth_name] = pd.Series(self.exp.resul[meth_name]['time'],
                                      name=meth_name)
        df = df.reindex_axis(df.mean().sort_values().index, axis=1)
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df)
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.ylabel('time in seconds')
        plt.show()

    def plot_hyper(self, figsize=(9, 3.5)):
        """ Plots cross-validation curves for all methods.
        Just enough to see if the grid was good. """
        local_methods = [key for key in self.exp.resul.keys()
                         if key not in ('dataset', 'hp',
                                        'metrics', 'task_metrics')]
        for method in local_methods:
            print('{}'.format(method))
            print('Best params')
            df = self.exp.resul[method]['hyper_params']
            params = [col for col in df.columns.values
                      if col not in ('fold', 'metric', 'val', 'tr')]
            if len(params) == 1:
                dfg = df.groupby(params)
                means = dfg.aggregate('mean')
                stds = dfg.aggregate('std')
                plt.figure()
                plt.plot(means['tr'].index, means['tr'], label='tr')
                plt.fill_between(means['tr'].index.values,
                                 means['tr'] - stds['tr'],
                                 means['tr'] + stds['tr'],
                                 color='gray', alpha=0.2)

                plt.plot(means['val'], label='val')
                plt.fill_between(means['val'].index.values,
                                 means['val'] - stds['val'],
                                 means['val'] + stds['val'],
                                 color='gray', alpha=0.2)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.xlabel('Error')
                plt.xlabel(params[0])
                plt.show()
            elif len(params) == 2:
                plt.figure()
                sns.lineplot(x=params[0], y='val', hue=params[1], data=df,
                             palette='Paired', markers=True)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.show()
            elif len(params) == 3:
                for lamb_3 in df[params[2]].unique():
                    local_df = df[df[params[2]] == lamb_3]
                    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)

                    for lamb_2 in local_df[params[1]].unique():
                        ldf = local_df[local_df[params[1]] == lamb_2]
                        dfg = ldf.groupby(params[0])
                        means = dfg.aggregate('mean')
                        stds = dfg.aggregate('std')

                        ax1.plot(means['tr'],
                                 label='{} = {:.4f}'.format(params[1], lamb_2))
                        ax1.fill_between(means['tr'].index.values,
                                         means['tr'] - stds['tr'],
                                         means['tr'] + stds['tr'],
                                         color='gray', alpha=0.2)

                        ax2.plot(means['val'],
                                 label='{} = {:.4f}'.format(params[1], lamb_2))
                        ax2.fill_between(means['val'].index.values,
                                         means['val'] - stds['val'],
                                         means['val'] + stds['val'],
                                         color='gray', alpha=0.2)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax1.set_title('TR {} = {:.4f}'.format(params[2], lamb_3))
                    ax2.set_title('VAL {} = {:.4f}'.format(params[2], lamb_3))
                    ax2.yaxis.set_tick_params(labelleft=True)
                    ax1.set_ylabel('error')
                    ax1.set_xlabel(params[0])
                    plt.show()

    def table_metric(self, metric='nmse'):
        """ Prints a table with all overall metrics per method
            (mean, std over all runs) using train and test set. """
        df = self.exp.resul['metrics']
        df = df[df['metric'] == metric]
        df = df.drop(['tr', 'run'], axis=1)
        df_g = df.groupby(['method'])
        display(df_g.agg([np.mean, np.std]))

    def table_metric_task(self, metric='mse'):
        """ (Real) Tabela da metrica por tarefa no conjunto which_set. """
        df = self.exp.resul['task_metrics']
        df = df[df['metric'] == metric]
        for task in df['task'].unique():
            local_df = df[df['task'] == task]
            local_df = local_df.drop(['tr', 'run'], axis=1)
            df_g = local_df.groupby(['method'])
            print('# Task {} #'.format(task))
            display(df_g.agg([np.mean, np.std]))

    def box_metric(self, metric='macc', which_set='te', y_log=False):
        """ (Real) Boxplot da metrica no conjunto which_set. """
        df = self.exp.resul['metrics']
        ax = sns.boxplot(data=df[df.metric == metric], x='method', y=which_set)
        if y_log:
            ax.set_yscale('log')
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.show()

    def bar_metric_task(self, metric='acc', which_set='te', figsize=None,
                        y_log=False, labels=None):
        """ (Real) Barplot da metrica por tarefa no conjunto which_set.

        Args:
            metric (str): metric to be plotted.
            which_set (str): 'te' or 'tr'.
            figsize (float, float): width, height in inches.
            y_log (bool): y axis in logscale.
        """
        df = self.exp.resul['task_metrics']
        df2 = df[df.metric == metric]
        plt.figure(figsize=figsize)
        ax = sns.barplot(data=df2, x='task', y=which_set, hue='method')
        ax.set_ylabel('error')
        if y_log:
            ax.set_yscale('log')
        if labels:
            ax.set_xticklabels(labels)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Error on {}'.format(which_set))
        plt.show()

    def heat_ws(self, figsize=None):
        """
        Heatmap of all parameter matrices W of all methods.

        Args:
            figsize (float, float): width, height in inches.
        """
        n_cols = len(self.exp.strategies.get_list())
        fig, axs = plt.subplots(1, n_cols, figsize=figsize)
        local_methods = [key for key in self.exp.resul
                         if key not in ('dataset', 'hp',
                                        'metrics', 'task_metrics')]
        for ind, method in enumerate(local_methods):
            w = self.exp.resul[method]['resul'][0]['W']
            mask = abs(w) < 10e-5
            sns.heatmap(w, mask=mask, ax=axs[ind], cmap=cmap_div, center=0, cbar=False)
            axs[ind].set_title('{}'.format(method))
            axs[ind].tick_params(
                axis='both',          # changes apply to both axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,         # ticks along the left edge are off
                right=False,         # ticks along the right edge are off
                labelbottom=False, # labels along the bottom edge are off
                labelleft=False) # labels along the bottom edge are off
        for ax in axs:
            for _, spine in ax.spines.items():
                spine.set_visible(True)
        fig.tight_layout()
        plt.show()

    def print_top_groups(self, k, method='GroupAMTL', figsize=None,
                         labels=None, kind='heatmap', save=False):
        """
        Heatmap of the k biggest relationship matrices, in terms of matricial
        norm1 of method.

        Args:
            k (int): top-k norm1 groups.
            method (str): method to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        norms = self._compute_b_norms(method)
        inds = np.argsort(norms)[-1*k:]
        if not labels:
            labels = ['TOTAL', 'T30', 'RECOG', 'MMSE', 'ADAS']
        for ind_g in inds:
            if kind == 'heatmap':
                self._print_group(ind_g, [method], figsize=figsize,
                                  labels=labels, save=save)
            elif kind == 'graph':
                self._print_group_graph(ind_g, [method], figsize=figsize, save=save)
            elif kind == 'hinton':
                run = 0
                Bs = self.exp.resul[method]['resul'][run]['Bs']
                fig, axs = plt.subplots(1, len(inds), figsize=figsize)
                for i, ind in inds:
                    self.hinton(Bs[ind_g].T, ax=axs[ind])
                    axs[i].set_xticklabels(labels)
                    axs[i].set_yticklabels(labels)
                    axs[i].tick_params(labelbottom=True, labelleft=True, labelrotation=.45)
                plt.plot()

    def print_groups(self, inds, methods, figsize=None, labels=None,
                     kind='heatmap'):
        """
        Heatmap of relationship matrices for all groups in inds.

        Args:
            inds (list of int): list with index of groups to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        for ind_g in inds:
            if kind == 'heatmap':
                self._print_group(ind_g, methods, figsize=figsize,
                                  labels=labels)
            elif kind == 'graph':
                self._print_group_graph(ind_g, methods, figsize=figsize,
                                        labels=labels)

    def _print_group(self, ind_g, methods=None, figsize=None, labels=None, save=False):
        """
        INTERNAL METHOD that prints ind_g group of methods.

        Args:
            ind_g (int): index of group to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        fold = 0
        for ind, method in enumerate(methods):
            B = self.exp.resul[method]['resul'][0]['Bs'][ind_g]
            print('{} fold {}: {}'.format(methods[ind], fold, ind_g))
            fig = plt.figure(figsize=figsize)
            ax = sns.heatmap(B, mask=abs(B) < 1e-4, center=0, cmap='gray_r',
                             annot=True, fmt='.2f', cbar=False)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.tick_params(labelbottom=True, labelleft=True, labelrotation=.45)
            for _, spine in ax.spines.items():
                spine.set_visible(True)

            plt.tight_layout()
            if save:
                plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_{}_bg{}.pdf'.format(method.lower(), ind_g))
            plt.show()

    def _compute_b_norms(self, method='GroupAMTL_nr'):
        """
        Computes the norm of all Bs over all fold.

        Args:
            method (str): method name

        Returns:
            norms (np.array): n_folds x n_groups. Each element f,g is
              the norm1 of Bg matrix in fold f.
            vals (np.array): shape (n_groups) with the mean of norm1
              of Bg matrix over all folds.
        """
        Bs = self.exp.resul[method]['resul'][0]['Bs']
        n_groups = len(Bs)
        norms = np.zeros(n_groups)
        for g in range(n_groups):
            norms[g] = sum(sum(abs(Bs[g])))
        return norms

    def _print_group_graph(self, ind_g, methods, figsize=None):
        """ INTERNAL METHOD that prints a graph for Bs[ind_g].

        Args:
            ind_g (int): index of group to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        fold = 0
        for ind, method in enumerate(methods):
            B = self.exp.resul[method]['resul'][0]['Bs'][ind_g]
            print('{} fold {}: {}'.format(methods[ind], fold, ind_g))
            fig = plt.figure(figsize=figsize)
            G = nx.DiGraph()
            for i in range(B.shape[0]):
                G.add_node(i)

            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    G.add_edge(i, j, weight=B[i, j])

            pos=nx.spring_layout(G) # positions for all nodes
            # nodes
            nx.draw_networkx_nodes(G, pos, node_size=700)
            # edges
            nx.draw_networkx_edges(G, pos, width=6)
            nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
            #plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_{}_bg{}.pdf'.format(method.lower(), ind_g))
            plt.axis('off')
            plt.show()

    def paper_struct(self, fold=0, method='MTRL', figsize=None):
        """ (Art) Heatmap dos Ws de todos os métodos do experimento na fold fold."""
        plt.figure(figsize=figsize)
        Omega = self.exp.resul['objs'][fold][method]['resul'][0]['Omega']
        mask = abs(Omega) < ZERO_PRECISION
        ax = sns.heatmap(Omega, mask=mask, cmap='gray', annot=True, fmt='.2f',
                         cbar=False)
        labels = ['TOTAL', 'T30', 'RECOG', 'MMSE', 'ADAS']
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.tick_params(labelbottom=True, labelleft=True, labelrotation=.45)
        #plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_{}.pdf'.format(method.lower()))
        plt.show()

    def paper_structures(self, inds, fold=0, figsize=None, labels=None):
        if not labels:
            labels = ['TOTAL', 'T30', 'RECOG', 'MMSE', 'ADAS']
        Omega = self.exp.resul['objs'][fold]['MTRL']['resul'][0]['Omega']
        mask = abs(Omega) < ZERO_PRECISION
        Omega[mask] = 0
        fig, ax = plt.subplots(len(inds) + 1, 1, figsize=figsize)
        self.hinton(matrix=Omega, ax=ax[0])
        for i, ind_g in enumerate(inds):
            Bg = self.exp.resul['objs'][fold]['GroupAMTL_nr']['resul'][0]['Bs'][ind_g]
            mask = abs(Bg) < ZERO_PRECISION
            Bg[mask] = 0
            self.hinton(matrix=Bg, ax=ax[i+1])
        ax[0].set_yticklabels(labels)
        ax[0].set_xticklabels([])
        ax[1].set_yticklabels(labels)
        ax[1].set_xticklabels([])
        ax[2].set_yticklabels(labels)
        ax[2].set_xticklabels(labels, rotation=45)

    def hinton(self, matrix, max_weight=None, ax=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        ax = ax if ax is not None else plt.gca()
        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('white')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=9))

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if abs(w) < ZERO_PRECISION else 'black'
            size = np.sqrt(np.abs(w) / max_weight) + 0.0001
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)
        ax.autoscale_view()
        ax.invert_yaxis()

    def sig_test(self, metric='nmse', reference='GroupAMTL'):
        """
        Computa Mann-Whitney rank test nas metricas gerais.
        Recomandado > 20 runs!!!
        Args:
            metric (str): metrica a ser utilizada na comparação
            reference (str): nome do método que sera comparado contra outros.
        """
        metrics = self.exp.resul['metrics']
        met = metrics[metrics['metric'] == metric]
        assert reference in met['method'].unique()
        x = met[met['method'] == reference]['te']
        methods = [met for met in met['method'].unique() if met != reference]
        res = pd.DataFrame()
        for method in methods:
            y = met[met['method'] == method]['te']
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(x, y, alternative='less')
            row = pd.Series([u, p, p < 0.05], index=['U', 'p-value', 'sig?'])
            res[method] = row
        return res.T


class PlotterVarM(Plotter):
    """
    """
    def __init__(self, exp):
        assert isinstance(exp, ExperimentVarParam)
        super().__init__(exp)

    def box_times(self, pos=0, figsize=None):
        local_methods = [key for key in self.exp.resul['objs'][pos].keys()
                         if key not in ('dataset', 'hp')]
        df = pd.DataFrame()
        for meth_name in local_methods:
            df[meth_name] = pd.Series(self.exp.resul['objs'][pos][meth_name]['time'],
                                      name=meth_name)
        df = df.reindex_axis(df.mean().sort_values().index, axis=1)
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df)
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.ylabel('time in seconds')
        plt.show()

    def plot_cross_val(self, pos=0, figsize=(9, 3.5)):
        """ Plots cross-validation curves for all methods.
        Just enough to see if the grid was good. """
        local_methods = [key for key in self.exp.resul['objs'][pos].keys()
                         if key not in ('dataset', 'hp')]
        for method in local_methods:
            print('{}'.format(method.upper()))
            print('Best params')
            df = self.exp.resul['objs'][pos][method]['hyper_params']
            params = [col for col in df.columns.values
                      if col not in ('metric', 'val', 'tr')]
            if len(params) == 1:
                dfg = df.groupby(params)
                means = dfg.aggregate('mean')
                stds = dfg.aggregate('std')
                plt.figure()
                plt.plot(means['tr'].index, means['tr'], label='tr')
                plt.fill_between(means['tr'].index.values,
                                 means['tr'] - stds['tr'],
                                 means['tr'] + stds['tr'],
                                 color='gray', alpha=0.2)

                plt.plot(means['val'], label='val')
                plt.fill_between(means['val'].index.values,
                                 means['val'] - stds['val'],
                                 means['val'] + stds['val'],
                                 color='gray', alpha=0.2)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.xlabel('Error')
                plt.xlabel(params[0])
                plt.show()
            elif len(params) == 2:
                for par_2 in df[params[1]].unique():
                    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
                    ldf = df[df[params[1]] == par_2]
                    dfg = df.groupby(params[0])
                    means = dfg.aggregate('mean')
                    stds = dfg.aggregate('std')
                    ax1.plot(means['tr'], label='{}: {}'.format(params[1], par_2))
                    ax1.fill_between(means['tr'].index.values,
                                     means['tr'] - stds['tr'],
                                     means['tr'] + stds['tr'],
                                     color='gray', alpha=0.2)
                    ax2.plot(means['val'], label='{}: {}'.format(params[1], par_2))
                    ax2.fill_between(means['val'].index.values,
                                     means['val'] - stds['val'],
                                     means['val'] + stds['val'],
                                     color='gray', alpha=0.2)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax1.set_title('TR {} = {:.4f}'.format(params[1], par_2))
                    ax2.set_title('VAL {} = {:.4f}'.format(params[1], par_2))
                    ax2.yaxis.set_tick_params(labelleft=True)
                    ax1.set_ylabel('error')
                    ax1.set_xlabel(params[0])
                    plt.show()
            elif len(params) == 3:
                for lamb_3 in df[params[2]].unique():
                    local_df = df[df[params[2]] == lamb_3]
                    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)

                    for lamb_2 in local_df[params[1]].unique():
                        ldf = local_df[local_df[params[1]] == lamb_2]
                        dfg = ldf.groupby(params[0])
                        means = dfg.aggregate('mean')
                        stds = dfg.aggregate('std')

                        ax1.plot(means['tr'],
                                 label='{} = {:.4f}'.format(params[1], lamb_2))
                        ax1.fill_between(means['tr'].index.values,
                                         means['tr'] - stds['tr'],
                                         means['tr'] + stds['tr'],
                                         color='gray', alpha=0.2)

                        ax2.plot(means['val'],
                                 label='{} = {:.4f}'.format(params[1], lamb_2))
                        ax2.fill_between(means['val'].index.values,
                                         means['val'] - stds['val'],
                                         means['val'] + stds['val'],
                                         color='gray', alpha=0.2)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax1.set_title('TR {} = {:.4f}'.format(params[2], lamb_3))
                    ax2.set_title('VAL {} = {:.4f}'.format(params[2], lamb_3))
                    ax2.yaxis.set_tick_params(labelleft=True)
                    ax1.set_ylabel('error')
                    ax1.set_xlabel(params[0])
                    plt.show()

    def line_metric(self, figsize=None, metric='macc', which_set='te',
                    divide=True, each=False, y_log=False):
        """ (Art) Lineplot da metrica no conjunto which_set. """
        if each:
            print('not implemented yet')
        else:
            df = self.exp.resul['metrics']
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
                    #plt.setp(ax.get_xticklabels(), rotation=45)
                    plt.show()

    def box_metric(self, metric='macc', which_set='te', y_log=False,
                   figsize=None):
        """ (Art) Boxplot da metrica no conjunto which_set. """
        df = self.exp.resul['metrics']
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df[df.metric == metric],
                         hue='method',
                         x='dataset_param',
                         y=which_set)
        if y_log:
            ax.set_yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def box_metric_per_val(self, param_val=np.Inf, metric='macc', which_set='te',
                           figsize=None, y_log=False):
        """ (Art) Boxplot da metrica no conjunto which_set variando dataset_param. """
        df = self.exp.resul['metrics']
        df2 = df[df.metric == metric]
        df3 = df2[df2.dataset_param == param_val]
        df_g = df3.groupby('method').agg(np.mean)
        inds = df_g.sort_values('te').index
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df3, x='method', y=which_set, order=inds)
        if y_log:
            ax.set_yscale('log')
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.show()

    def bar_metric_task(self, param_val=np.Inf, metric='acc', which_set='te',
                        figsize=None, y_log=False, methods=None):
        """ (Art) Barplot da metrica por tarefa no conjunto which_set no param_val. """
        df = self.exp.resul['task_metrics']
        df2 = df[df.metric == metric]
        df3 = df2[df2.dataset_param == param_val]
        if methods:
            df3 = df3[df3['method'].isin(methods)]
        plt.figure(figsize=figsize)
        ax = sns.barplot(data=df3, x='task', y=which_set, hue='method')
        if y_log:
            ax.set_yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def heat_ws(self, pos=0, figsize=None):
        """ (Art) Heatmap dos Ws de todos os métodos do experimento na posicao pos."""
        if hasattr(self.exp.resul['objs'][pos]['dataset'], 'W'):
            w = self.exp.resul['objs'][pos]['dataset'].W
            mask = abs(w) < ZERO_PRECISION
            plt.figure(figsize=figsize)
            sns.heatmap(w, mask=mask, cmap=cmap_div, center=0)
            plt.title('Original')
            plt.show()
        local_methods = [key for key in self.exp.resul['objs'][pos]
                         if key not in ('dataset', 'hp')]
        for method in local_methods:
            w = self.exp.resul['objs'][pos][method]['resul'][0]['W']
            mask = abs(w) < ZERO_PRECISION
            plt.figure(figsize=figsize)
            sns.heatmap(w, mask=mask, cmap=cmap_div, center=0)
            plt.title('{}'.format(method))
            plt.show()

    def print_groups(self, inds, pos=0, methods=None, figsize=None,
                     kind='heatmap'):
        """
        Heatmap of relationship matrices for all groups in inds.

        Args:
            inds (list of int): list with index of groups to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        for ind_g in inds:
            if kind == 'heatmap':
                self._print_group(ind_g, pos, methods, figsize=figsize)
            elif kind == 'graph':
                self._print_group_graph(ind_g, methods, figsize=figsize)

    def _print_group(self, ind_g, pos=0, methods=None, figsize=None):
        """
        INTERNAL METHOD that prints ind_g group of methods.

        Args:
            ind_g (int): index of group to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        run = 0
        for ind, method in enumerate(methods):
            B = self.exp.resul['objs'][pos][method]['resul'][run]['Bs'][ind_g]
            print('{} group {}'.format(methods[ind], ind_g))
            plt.figure(figsize=figsize)
            ax = sns.heatmap(B, mask=abs(B) < 1e-4, center=0, cmap='gray_r',
                             annot=True, fmt='.2f', cbar=False)
            ax.tick_params(labelbottom=True, labelleft=True, labelrotation=.45)

            for _, spine in ax.spines.items():
                spine.set_visible(True)
            #plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_{}_bg{}.pdf'.format(method.lower(), ind_g))
            plt.show()

    def _compute_b_norms(self, pos=0, method='GroupAMTL_nr'):
        """
        Computes the norm of all Bs over all folds.

        Args:
            method (str): method name

        Returns:
            norms (np.array): n_folds x n_groups. Each element f,g is
              the norm1 of Bg matrix in fold f.
            vals (np.array): shape (n_groups) with the mean of norm1
              of Bg matrix over all folds.
        """
        run = 0
        Bs = self.exp.resul['objs'][pos][method]['resul'][run]['Bs']
        n_groups = len(Bs)
        norms = np.zeros(n_groups)
        for g in range(n_groups):
            norms[g] = sum(sum(abs(Bs[g])))
        return norms

    def _print_group_graph(self, ind_g, methods, figsize=None):
        """ INTERNAL METHOD that prints a graph for Bs[ind_g].

        Args:
            ind_g (int): index of group to plot.
            methods (list of str): list of method names to be plotted.
            figsize (float, float): width, height in inches.
            labels (list of str or None): list with labels for all tasks.
        """
        fold = 0
        for ind, method in enumerate(methods):
            B = self.exp.resul[method]['resul'][0]['Bs'][ind_g]
            print('{} fold {}: {}'.format(methods[ind], fold, ind_g))
            plt.figure(figsize=figsize)
            G = nx.DiGraph()
            for i in range(B.shape[0]):
                G.add_node(i)

            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    G.add_edge(i, j, weight=B[i, j])

            pos = nx.spring_layout(G) # positions for all nodes
            # nodes
            nx.draw_networkx_nodes(G, pos, node_size=700)
            # edges
            nx.draw_networkx_edges(G, pos, width=6)
            nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
            #plt.savefig('/home/churros/pasta/codes/gamtl/figs/adni_{}_bg{}.pdf'.format(method.lower(), ind_g))
            plt.axis('off')
            plt.show()

    def line_metric_specific(self, methods, figsize=None,
                             metric='nmse', which_set='te',
                             y_log=False):
        """ (Art) Lineplot da metrica no conjunto which_set. """
        df = self.exp.resul['metrics']
        df = df[df['method'].isin(methods)]
        df = df[df['metric'] == metric]
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

        #methods = df.method.unique()
        for method in methods:
            dfg = df[df['method'] == method].groupby('dataset_param')
            means = dfg.aggregate('mean')
            stds = dfg.aggregate('std')
            plt.figure()
            plt.plot(means['tr'].index, means['tr'], label='tr')
            plt.fill_between(means['tr'].index.values,
                             means['tr'] - stds['tr'],
                             means['tr'] + stds['tr'],
                             color='gray', alpha=0.2)

            plt.plot(means['te'], label='val')
            plt.fill_between(means['te'].index.values,
                             means['te'] - stds['te'],
                             means['te'] + stds['te'],
                             color='gray', alpha=0.2)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.xlabel('Error')
            plt.xlabel('number of samples')
            plt.title(method)
            plt.show()
 
    def paper_nmse(self, methods, labels, figsize=None, y_log=False, save=False, x_until=130):
        """ (Art) Lineplot da metrica no conjunto which_set. """
        # tem 10 aqui...
        markers = ['D', 's', 'o', 'x', '>', 's', 'X', 'P', 'D', '*']
        df = self.exp.resul['metrics']
        df = df[df['method'].isin(methods)]
        df = df[df['metric'] == 'nmse']
        df = df[df['dataset_param'] <= x_until]
        # plota tudo primeiro
        plt.figure(figsize=figsize)
        for i, method in enumerate(methods):
            dfg = df[df['method'] == method].groupby('dataset_param')
            means = dfg.aggregate('mean')
            stds = dfg.aggregate('std')
            ax = plt.plot(means['te'].index, means['te'], label='te',
                          marker=markers[i], markersize=8, color='black',
                          linestyle='dashed')
            plt.fill_between(means['te'].index.values,
                             means['te'] - stds['te'],
                             means['te'] + stds['te'],
                             color='gray', alpha=0.2)
            #ax = sns.lineplot(data=df[df['method'] == method],
            #                  x='dataset_param', y='te', marker=markers[i],
            #                  markersize=2, linestyle='-', color='b')
        plt.ylabel('NMSE')
        if y_log:
            plt.yscale('log')
            plt.ylabel('log of NMSE')
        plt.xticks(df['dataset_param'].unique())
        plt.xlabel('Number of Samples')
        #plt.grid()
        plt.legend(labels, loc='upper right', frameon=True)
        plt.tight_layout()
        if save:
            plt.savefig('/home/churros/pasta/codes/gamtl/figs/{}_nmse.pdf'.format(self.exp.filename[8:].split('_')[0]))
        plt.show()

    def paper_hinton_bs(self, method, pos, max_weight=None, figsize=None, save=False, run=0):
        """
        (Art) Hintonmap dos Ws de todos os métodos do experimento na posicao pos.
        Code from: https://matplotlib.org/gallery/specialty_plots/hinton_demo.html
        """
        fig, axs = plt.subplots(nrows=2, ncols=len(pos)+1, figsize=figsize)
        # dataset
        Bs = self.exp.resul['objs'][run]['dataset'].Bs

        self.hinton(Bs[0].T, max_weight=max_weight, ax=axs[0, 0])
        self.hinton(Bs[1].T, max_weight=max_weight, ax=axs[1, 0])
        axs[0, 0].tick_params(
            axis='both',          # changes apply to both axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,         # ticks along the left edge are off
            right=False,         # ticks along the right edge are off
            labelbottom=False, # labels along the bottom edge are off
            labelleft=False) # labels along the bottom edge are off

        axs[0, 0].set_title('B of group 1')
        axs[1, 0].set_title('B of group 2')
        axs[1, 0].set_ylabel('from task')
        axs[1, 0].set_xlabel('to task')
        axs[1, 0].set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])
        axs[1, 0].set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # gamtl
        for ind, val in enumerate(pos):
            Bs = self.exp.resul['objs'][val][method]['resul'][0]['Bs']
            self.hinton(Bs[0].T, max_weight=max_weight, ax=axs[0, ind+1])
            self.hinton(Bs[1].T, max_weight=max_weight, ax=axs[1, ind+1])
            for i in [0, 1]:
                axs[i, ind+1].tick_params(
                    axis='both',          # changes apply to both axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    left=False,         # ticks along the left edge are off
                    right=False,         # ticks along the right edge are off
                    labelbottom=False, # labels along the bottom edge are off
                    labelleft=False) # labels along the bottom edge are off
            axs[0, ind+i].set_title('m: {}'.format(self.exp.dataset_params[val]))

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        # fig.suptitle(method, size=14)

        if save:
            plt.savefig('/home/churros/pasta/codes/gamtl/figs/{}_hinton_bs.pdf'.format(self.exp.filename[8:].split('_')[0]))
        plt.show()

    def paper_ws(self, pos, methods=None, figsize=None, save=False):
        """
        """
        if not methods:
            methods = [i for i in self.exp.resul['objs'][0].keys()\
                       if i not in ('dataset', 'hp')]
        qtd_methods = len(methods)
        fig, axs = plt.subplots(nrows=qtd_methods+1, ncols=len(pos),
                                figsize=figsize)
        tick_params = {
            'axis': 'both',        # changes apply to both axis
            'which': 'both',       # both major and minor ticks are affected
            'bottom': False,       # ticks along the bottom edge are off
            'top': False,          # ticks along the top edge are off
            'left': False,         # ticks along the left edge are off
            'right': False,        # ticks along the right edge are off
            'labelbottom': False,  # labels along the bottom edge are off
            'labelleft': False     # labels along the bottom edge are off
        }
        for i, val in enumerate(pos):
            w = self.exp.resul['objs'][val]['dataset'].W
            vmin = np.min(w)
            vmax = np.max(w)
            mask = abs(w) < ZERO_PRECISION
            sns.heatmap(w, mask=mask, cmap=cmap_div, center=0, ax=axs[0, i],
                        vmin=vmin, vmax=vmax, cbar=False)
            axs[0, i].tick_params(**tick_params)

            for j, method in enumerate(methods):
                w = self.exp.resul['objs'][i][method]['resul'][0]['W']
                mask = abs(w) < ZERO_PRECISION
                sns.heatmap(w, mask=mask, cmap=cmap_div, center=0,
                            ax=axs[j+1, i], vmin=vmin, vmax=vmax, cbar=False)
                axs[j+1, i].tick_params(**tick_params)
            axs[0, i].set_title('m = {}'.format(self.exp.dataset_params[i]))
        axs[0, 0].set_ylabel('Original')
        for j, method in enumerate(methods):
            axs[j+1, 0].set_ylabel(method)
        if save:
            plt.savefig('/home/churros/pasta/codes/gamtl/figs/art_ws.pdf')
        plt.plot()

    def hinton(self, matrix, max_weight=None, ax=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        ax = ax if ax is not None else plt.gca()
        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('white')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=9))

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if abs(w) < ZERO_PRECISION else 'black'
            size = np.sqrt(np.abs(w) / max_weight) + 0.0001
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)
        ax.autoscale_view()
        ax.invert_yaxis()
