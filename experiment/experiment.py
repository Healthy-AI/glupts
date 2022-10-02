import random
import git
import datetime
import os
import numpy as np
import pandas as pd
import warnings
import math
from data.dataset import SampleSizeError
from itertools import product
import matplotlib.pyplot as plt
import multiprocessing
import time
from helpers import SVCCA
import subprocess
import pkg_resources
import compress_pickle
import json

BIAS_VAR_SEED = 500
BASH_TEMPLATES = {'gpu': 'gpu_slurm.sh'}


def execute_jobs(inQ, outQ):
    """
    Gets called by a single worker, gets jobs from the input queue, processes them and pushes the result back to the
    output queue
    :param args:
    """
    # get jobs from the queue
    while True:
        try:
            config = inQ.get(block=False)
        except Exception as e:
            continue

        if config == 'DONE':
            break
        # unpack args
        data_inst, seq_len, seq_step, hp_tuning, model, score_fun, \
        random_seed, sample_size, seq_gap, test_ratio, folder_path, bias_var_seed = config
        # compute results
        results = single_job(model, random_seed, score_fun, seq_gap, test_ratio,
                             data_inst, seq_len, seq_step, sample_size, hp_tuning, folder_path, bias_var_seed)

        outQ.put(results, timeout=0.1)


def single_job(model, random_seed, score_fun, seq_gap, test_ratio, data_obj, seq_len, seq_step_size, n, hp_tuning,
               folder_path, bias_var_seed):
    """
    Takes a job configuration and computes the results for it, gets called by execute_jobs
    """

    # catch SampleSizeError in case there are not enough samples for the desired configuration
    try:
        train, test = data_obj.get_train_test_data(seq_len, seq_step_size, test_ratio, n, random_seed)
    except SampleSizeError as sse:
        warnings.warn('Sample size not reached. Skipping run.')
        return sse

    # unpack the data
    train_x, train_y = train
    test_x, test_y = test

    if hp_tuning:
        model.hyperparameter_search(train_x, train_y, random_seed=random_seed)
    model.fit(train_x, train_y, dataset=data_obj, savepath=folder_path)
    y_hat = model.predict(test_x, dataset=data_obj)

    score = score_fun.score(y_hat, test_y)

    # store result
    result_dict = {
        'Data': data_obj.name,
        'Time': datetime.datetime.now(datetime.timezone.utc).now().strftime(
            '%m/%d/%Y, %H:%M:%S'),
        'Model_Number': model.model_id,
        'Sequence_Length': seq_len,
        'Sequence_Step': seq_step_size,
        'Sequence_Gap': seq_gap,
        'Target_Columns': str(data_obj.target_cols),
        'Sample_Size': n,
        'Seed': random_seed,
        'Model_Name': model.name,
    }

    synthetic_system = hasattr(data_obj, 'get_latent_vars')
    if synthetic_system:
        lin_sim = compute_linear_similarity(data_obj, model, test_x)
        if lin_sim is not None:
            result_dict['SVCCA_Score'] = lin_sim[0]
            result_dict['CCA_Score'] = lin_sim[1]
        _, (test_x, _) = data_obj.get_train_test_data(seq_len, 1, 0.5, 1000, bias_var_seed)
        bias_variance_results = {'Predictions': model.predict(test_x),
                                 'Sequence_Length': seq_len,
                                 'Sequence_Step': seq_step_size,
                                 'Sample_Size': n,
                                 'Seed': random_seed,
                                 'System_Seed': data_obj.system_seed,
                                 'Model_Name': model.name,
                                 'Model_Number': model.model_id,
                                 'Data': data_obj.name
                                 }
    else:
        bias_variance_results = None

    result_dict = {**result_dict, **model.get_info(), **score, **data_obj.get_info()}
    return result_dict, bias_variance_results


def compute_linear_similarity(data_obj, model_obj, test_set_x):
    # if data set has known latent variables compute the linear similarity
    latent_method_data = getattr(data_obj, 'get_latent_vars', None)
    latent_method_model = getattr(model_obj, 'predict_latent', None)
    if (callable(latent_method_data)) and (callable(latent_method_model)):
        latent_pred = latent_method_model(test_set_x)
        latent_true = latent_method_data()[1]
        svcca_coefs = []
        cca_coefs = []
        for t in range(latent_true.shape[1]):
            svcca_coefs.append(SVCCA(latent_pred[:, t], latent_true[:, t], use_PCA=True)[0])
            cca_coefs.append(SVCCA(latent_pred[:, t], latent_true[:, t], use_PCA=False)[0])
        svcca_mean = float(np.mean(svcca_coefs))
        cca_mean = float(np.mean(cca_coefs))
        return svcca_mean, cca_mean
    else:
        return None


def single_node_parallel_job(configs, workers, csv_path=None):
    inQ = multiprocessing.Queue()
    outQ = multiprocessing.Queue()

    workers = min(multiprocessing.cpu_count(), workers)
    processes = [multiprocessing.Process(target=execute_jobs, args=(inQ, outQ)) for _ in range(workers)]
    number_of_runs = len(configs)

    for config in configs:
        inQ.put(config)
    for _ in processes:
        inQ.put('DONE')

    results = []
    bias_var_results = []
    errors = []

    print(f'Starting {workers} processes for {number_of_runs} runs.')

    for p in processes:
        p.start()

    received_counter = 0
    while True:
        alive = any(p.is_alive() for p in processes)
        while not outQ.empty():
            try:
                result = outQ.get(block=False)
                received_counter += 1
                if isinstance(result, tuple):
                    sample_eff, bias_var_res = result
                    results.append(sample_eff)
                    if bias_var_res is not None:
                        bias_var_results.append(bias_var_res)
                else:
                    errors.append(result)
                print(f'Progress: {round(received_counter / number_of_runs * 100, 2)} %')
            except Exception as e:
                pass
        if not alive:
            break
    for p in processes:
        p.join()

    inQ.close()
    outQ.close()

    results_df = pd.DataFrame(results)
    if csv_path is None:
        return results_df, bias_var_results, errors
    else:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path)
        filename = os.path.basename(os.path.normpath(csv_path)).replace('.csv', '.gz')
        root = os.path.dirname(os.path.dirname(csv_path))
        dir_bias_var = os.path.join(root, 'Bias_Var')
        os.makedirs(dir_bias_var, exist_ok=True)
        filepath = os.path.join(dir_bias_var, filename)
        with open(filepath, 'wb') as f:
            compress_pickle.dump(bias_var_results, f)


class experiment():
    default_x_axes = ['Sequence_Length', 'Sequence_Step', 'Sample_Size']
    viz_group_by_cols = ['Data', 'Sample_Size', 'Sequence_Step', 'Sequence_Length']
    ignore_columns_when_aggregating = ['Time', 'Seed', 'State_Dict', 'Model_Path']
    optional_score_columns = ['SVCCA_Score', 'CCA_Score']

    @classmethod
    def from_path(self, path):
        ex = experiment([], [], [], [], [], 0, 0, 1, 1, None)
        self.save_dir = path
        path = os.path.join(path, 'experiment.gz')
        print(f'Loading experiment file: {path}')
        ex.load(path)
        return ex

    def __init__(self,
                 data_instances,
                 models,
                 sample_sizes,
                 seq_lengths,
                 seq_steps,
                 seq_gap,
                 test_ratio,
                 random_seed,
                 repetitions,
                 score_function,
                 hp_tuning=True):
        self.data_instances = data_instances

        self.models = models
        model_num = 0
        for m in self.models:
            m.model_id = model_num
            model_num += 1
        self.bias_var_seed = BIAS_VAR_SEED
        self.sample_sizes = sample_sizes
        self.seq_lengths = seq_lengths
        self.seq_steps = seq_steps
        self.seq_gap = seq_gap
        self.test_ratio = test_ratio
        self.repetitions = repetitions
        self.score_fun = score_function
        self.random_seed_start = random_seed
        np.random.seed(random_seed)
        self.random_seed_sequence = np.random.randint(0, 2 ** 16, repetitions).tolist()
        self.next_seed = 0
        self.creation_time = datetime.datetime.now(datetime.timezone.utc)
        self.hp_tuning = hp_tuning

        # modify these from the outside to use different settings
        self.do_save = True
        self.verbose = True
        self.remember_git_info()

    def run_single_node(self, savepath, workers):
        run_configs, start_system_time = self.prepare_run(savepath)
        number_of_runs = len(run_configs)

        self.results, self.bias_var_results, errors = single_node_parallel_job(run_configs, workers)

        # Save results and self in case post processing fails
        if self.do_save:
            self.save()

        if self.results.shape[0] > 0:
            self.compute_aggregate_scores()
        else:
            raise Exception('Result Dataframe is empty.')

        # Print how long the computation took
        sec_tot = time.time() - start_system_time
        hours = int(sec_tot / 3600)
        minutes = int((sec_tot % 3600) / 60)
        secs = int(sec_tot % 60)
        self.print(f'\nComputing Time: {hours} h {minutes} min {secs} sec')

        # Report back all errors
        self.print('\n\nErrors:')
        for e in errors:
            self.print(e)
        print(f'{len(errors)} total errors out of {number_of_runs} runs in total. That is '
              f'{round(len(errors) / number_of_runs * 100, 2)} %.')

        # Save results and self
        if self.do_save:
            self.save()

    def prepare_run(self, savepath):
        # create directories for run
        self.run_start_time = datetime.datetime.now(datetime.timezone.utc)
        start_system_time = time.time()
        savepath = os.path.abspath(savepath)
        self.save_dir = os.path.join(savepath, self.run_start_time.strftime('%Y_%m_%d_%H_%M_%S'))

        # save run object at start
        if self.do_save:
            self.save()

        for data_obj in self.data_instances:
            data_obj.prepare_for_experiment(self.seq_lengths, self.seq_steps, self.seq_gap, self.sample_sizes)

        run_configs = list(product(self.data_instances, self.seq_lengths, self.seq_steps, [self.hp_tuning], self.models,
                                   [self.score_fun], self.random_seed_sequence, self.sample_sizes, [self.seq_gap],
                                   [self.test_ratio], [self.save_dir], [self.bias_var_seed]))
        assert len(run_configs) > 0, 'Incomplete configuration.'

        random.shuffle(run_configs)
        return run_configs, start_system_time

    def run_slurm_jobs(self, savepath, workers_per_node, nodes, hours_per_node, jobname, template='test_bash'):
        run_configs, start_system_time = self.prepare_run(savepath)
        number_of_runs = len(run_configs)
        runs_per_node = math.ceil(number_of_runs / nodes)

        batches = []
        for n in range(nodes):
            batches.append(run_configs[n * runs_per_node:(n + 1) * runs_per_node])
        assert sum([len(b) for b in batches]) == number_of_runs

        self.config_path = os.path.join(self.save_dir, '_config_pickles')
        self.config_path_all = []
        os.makedirs(self.config_path)
        for n, batch in enumerate(batches):
            filepath = os.path.join(self.config_path, str(n).zfill(4) + '.gz')
            with open(filepath, 'wb') as f:
                compress_pickle.dump(batch, f)
            self.config_path_all.append(filepath)

        self.create_slurm_job_files(nodes, self.config_path_all, hours_per_node, workers_per_node, jobname, template)
        if template == 'test':
            test = True
        else:
            test = False
        self.submit_slurm_jobs(test=test)

    def submit_slurm_jobs(self, test=False):
        assert hasattr(self, 'job_file_paths_all'), 'No job files have been created'
        for job in self.job_file_paths_all:
            subprocess.run(['chmod', '+x', job])
            if test:
                subprocess.call(['bash', job])
            else:
                subprocess.Popen(['sbatch', job])

    def create_slurm_job_files(self, nodes, job_config_filepaths, hours_per_node, workers, jobname, template):

        bash_file = pkg_resources.resource_filename('experiment', BASH_TEMPLATES[template])
        with open(bash_file, 'r') as f:
            header = f.read()

        # set up the directories
        self.job_file_path = os.path.join(self.save_dir, 'slurm_jobs')
        os.makedirs(self.job_file_path)
        os.makedirs(os.path.join(self.job_file_path, '_out'))
        os.makedirs(os.path.join(self.job_file_path, '_error'))

        # make all the bash scripts
        self.job_file_paths_all = []
        for node, conf_path in zip(range(nodes), job_config_filepaths):
            job_path = os.path.join(self.job_file_path, str(node).zfill(4) + '.sh')
            out_path = os.path.join(self.job_file_path, '_out', str(node).zfill(4) + '.txt')
            e_path = os.path.join(self.job_file_path, '_error', str(node).zfill(4) + '.txt')
            csv_path = os.path.join(self.save_dir, 'results', str(node).zfill(4) + '.csv')
            with open(job_path, 'x') as f:
                f.write(header % (jobname, out_path, e_path, workers, hours_per_node, hours_per_node, \
                                  conf_path, workers, csv_path))
            self.job_file_paths_all.append(job_path)

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)

        with open(os.path.join(self.save_dir, 'experiment.gz'), 'wb') as f:
            blacklist = ['results', 'bias_var_results', 'aggregate_results']
            save_dict = {key: item for key, item in self.__dict__.items() if key not in blacklist}
            compress_pickle.dump(save_dict, f)
        self.print('Saved experiment file.')

        if hasattr(self, 'results'):
            os.makedirs(os.path.join(self.save_dir, 'results'), exist_ok=True)
            self.results.to_csv(os.path.join(self.save_dir, 'results', 'results.csv'))
            self.print('Saved results.')

        if hasattr(self, 'bias_var_results'):
            folderpath = os.path.join(self.save_dir, 'Bias_Var')
            os.makedirs(folderpath, exist_ok=True)
            with open(os.path.join(folderpath, 'bias_var_res.gz'), 'wb') as f:
                compress_pickle.dump(self.bias_var_results, f)

        if hasattr(self, 'aggregate_results'):
            self.aggregate_results.to_csv(os.path.join(self.save_dir, 'aggregate_results.csv'))
            self.print('Saved aggregate results.')

    def load(self, path):
        with open(path, 'rb') as f:
            attributes = compress_pickle.load(f)
            if 'save_dir' in attributes:
                del attributes['save_dir']
            self.__dict__ = attributes

        path = os.path.join(self.save_dir, 'results', 'results.csv')
        if os.path.exists(path):
            self.results = pd.read_csv(path).iloc[:, 1:]

        path = os.path.join(self.save_dir, 'Bias_Var', 'bias_var_res.gz')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.bias_var_results = compress_pickle.load(f)

        path = os.path.join(self.save_dir, 'aggregate_results.csv')
        if os.path.exists(path):
            self.aggregate_results = pd.read_csv(path).iloc[:, 1:]

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def merge_slurm_jobs(self):

        # sample efficiency CSV files
        resultpath = os.path.join(self.save_dir, 'results')
        resultfiles = [f for f in os.listdir(resultpath) if '.csv' in f and f != 'results.csv']
        file = resultfiles.pop(0)
        df = pd.read_csv(os.path.join(self.save_dir, 'results', file)).iloc[:, 1:]
        for f in resultfiles:
            file = os.path.join(self.save_dir, 'results', f)
            new_df = pd.read_csv(file).iloc[:, 1:]
            df = pd.concat((df, new_df), axis=0, ignore_index=True).reset_index(inplace=False, drop=True)
        self.results = df

        # bias variance pickles
        resultpath = os.path.join(self.save_dir, 'Bias_Var')
        resultfiles = [f for f in os.listdir(resultpath) if '.gz' in f and f != 'results.gz']
        bias_variance_results = []
        for f in resultfiles:
            with open(os.path.join(resultpath, f), 'rb') as f:
                bias_variance_results += compress_pickle.load(f)
        self.bias_var_results = bias_variance_results
        self.save()

    def compute_aggregate_scores(self):
        # goal is to create a groupby object to group by all the columns of interest to determine the different aggregate results
        columns_groupby = set(self.results.columns.tolist())

        score_cols = self.score_fun.get_score_cols() + [c for c in experiment.optional_score_columns if
                                                        c in columns_groupby]
        # remove the score columns and others that we don't want to group by
        for c in experiment.ignore_columns_when_aggregating + score_cols:
            if c in columns_groupby:
                columns_groupby.remove(c)
        columns_groupby = list(columns_groupby)
        hyperparams = self.get_all_hyperparams_used()
        columns_groupby = [c for c in columns_groupby if c not in hyperparams]

        # groupby object
        groups = self.results.groupby(columns_groupby, dropna=False)
        # calculate means and stds
        means = groups[score_cols].mean().reset_index()
        means = means.rename(columns={name: name + '_Mean' for name in score_cols})
        std = groups[score_cols].std().reset_index()
        std_cols = {name: name + '_Std' for name in score_cols}
        std = std.rename(columns=std_cols)
        std_cols = list(std_cols.values())
        means.loc[:, std_cols] = std.loc[:, std_cols]
        # save entire dataframe
        self.aggregate_results = means
        self.calculate_bias_var_results()

    def remember_git_info(self):
        try:
            repo = git.Repo(search_parent_directories=True)
            self.git_info = {'repo': repo,
                             'hash': repo.head.object.hexsha,
                             'branch': repo.active_branch}
        except Exception as e:
            warnings.warn('Getting git info failed: ' + str(e))

    def visualize_scores(self, *args, **kwargs):
        assert hasattr(self, 'aggregate_results'), 'Cannot visualize results when there are none.'
        metrics = self.score_fun.get_score_cols() + [c for c in experiment.optional_score_columns if
                                                     c in self.results.columns.tolist()]
        df = self.aggregate_results

        for metric in metrics:
            df[metric + '_Mean+Std'] = df[metric + '_Mean'] + df[metric + '_Std']
            df[metric + '_Mean-Std'] = df[metric + '_Mean'] - df[metric + '_Std']

        # iterate over the y-axis variables
        for metric in metrics:

            # which type of x-axis variables are there, iterate over those
            for x_axis_label in experiment.default_x_axes:
                # find out how many plots to do and group the data accordingly
                groupby_cols = [c for c in experiment.viz_group_by_cols if c != x_axis_label]
                gb = df.groupby(groupby_cols)

                # iterate over the different groups that need a plot for this x-label
                for group_props, idx in gb.groups.items():

                    temp_df = df.loc[idx, :]

                    # now group again by model
                    model_gb = temp_df.groupby(['Model_Number'])

                    # z-order
                    z = 2

                    data_found = False
                    if "/Library/TeX/texbin" not in os.environ['PATH']:
                        os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'

                    rc = {"font.family": "Times New Roman",
                          "mathtext.fontset": "cm"}
                    plt.rcParams.update(rc)

                    fig = plt.figure(figsize=(5, 5))
                    for model_num, modx in model_gb.groups.items():
                        if len(modx) == 1:
                            continue
                        else:
                            data_found = True
                        # create a plot here

                        # get color of the corresponding model
                        color = self.models[model_num].get_color()
                        name = self.models[model_num].name
                        if np.any(~np.isnan(temp_df.loc[modx, metric + '_Mean'])):
                            if 'marker_map' in kwargs:
                                if name in kwargs['marker_map']:
                                    marker = kwargs['marker_map'][name]
                            else:
                                marker = 'o'

                            plt.fill_between(temp_df.loc[modx, x_axis_label],
                                             temp_df.loc[modx, metric + '_Mean-Std'],
                                             temp_df.loc[modx, metric + '_Mean+Std'],
                                             color=color, alpha=0.1, zorder=z)
                            z += 1

                            plt.plot(temp_df.loc[modx, x_axis_label], temp_df.loc[modx, metric + '_Mean'],
                                     marker + '-', c=color, label=name, linewidth=1.2, zorder=100 + z)

                            z += 1


                    if 'special_point' in kwargs:
                        plt.scatter(kwargs['special_point'][0], kwargs['special_point'][1], s=100, marker="*",
                                    c=kwargs['special_point'][2], zorder=100 + z)

                    if data_found:
                        if 'fontsize' in kwargs:
                            fontsize = kwargs['fontsize']
                        else:
                            fontsize = 12
                        plt.ylabel(metric.replace('_', ' ').replace('R2', 'R²'), fontsize=fontsize)
                        plt.xlabel(x_axis_label.replace('_', ' ').replace('Sample Size', 'Sample Size, $m$'),
                                   fontsize=fontsize)
                        plt.xticks(fontsize=fontsize)
                        plt.yticks(fontsize=fontsize)
                        # plot settings

                        # LEGEND
                        if 'legend' in kwargs:
                            use_legend = kwargs['legend']
                        else:
                            use_legend = True
                        if use_legend:
                            if 'legend_order' in kwargs:
                                legend_order = kwargs['legend_order']
                                handles, labels = plt.gca().get_legend_handles_labels()
                                legend_order = [l for l in legend_order if l in labels]
                                current_order = {l: h for l, h in zip(labels, handles)}
                                reordered_handles = [current_order[l] for l in legend_order]
                                leg = plt.legend(reordered_handles, legend_order, prop={'size': 12})
                            else:
                                leg = plt.legend(prop={'size': 12})
                            leg.set_zorder(z+101)
                        ax = plt.gca()
                        # ax.set_yticks(np.arange(-1.1, 1.1, 0.1))
                        # ax.set_yticks(np.arange(-1.1, 1.1, 0.05), minor=True)

                        if 'ylim' in kwargs:
                            plt.ylim(kwargs['ylim'])

                        # make title
                        if 'title' in kwargs:
                            plot_title = bool(kwargs['title'])
                        else:
                            plot_title = True

                        if plot_title:
                            title = ''
                            for k, v in zip(groupby_cols, group_props):
                                title += str(k) + ': ' + str(v) + '    '
                            title = title[:-4]
                            plt.title(title, fontsize=7)

                        path = os.path.join(self.save_dir, 'Visuals', metric)
                        os.makedirs(path, exist_ok=True)
                        path = os.path.join(path, x_axis_label)
                        os.makedirs(path, exist_ok=True)
                        filename = str(group_props).replace(',', '').replace('(', '').replace(')', '').replace(' ', '_')
                        path = os.path.join(path, filename + '.pdf')
                        plt.grid(which='minor', zorder=1, alpha=0.1)
                        plt.grid(which='major', zorder=2, alpha=0.2)
                        plt.tight_layout()
                        plt.savefig(path, format='pdf', facecolor='#ffffff')
                    plt.close()

    def calculate_bias_var_results(self):
        if not hasattr(self, 'bias_var_results'):
            ex.merge_slurm_jobs()

        if not hasattr(self, 'bias_var_results'):
            return None

        data_instance_names = [d.name for d in self.data_instances]
        assert max(
            [data_instance_names.count(d) for d in data_instance_names]) == 1, 'Duplicate data instance names found!'

        def get_mse(prediction, true):
            return np.mean(np.sum(np.power(prediction - true, 2), -1), axis=0)

        predictions = {}

        fields = ['Sequence_Length', 'Model_Name', 'Model_Number', 'Sample_Size', 'Data', 'Seed']
        seeds = set()
        for res in self.bias_var_results:
            key = tuple(res[f] for f in fields)
            predictions[key] = res['Predictions']
            seeds.add(res['Seed'])

        result_data = []
        for seq_len in self.seq_lengths:
            for model in self.models:
                # only data instances that are synthetic
                for d in [k for k in self.data_instances if hasattr(k, 'get_latent_vars')]:
                    for n in self.sample_sizes:
                        yhats = [predictions[seq_len, model.name, model.model_id, n, d.name, s] for s in seeds]
                        mean_prediction = np.mean(np.stack(yhats, -1), -1)
                        _, (_, actual_y) = d.get_train_test_data(seq_len, 1, 0.5, 1000, self.bias_var_seed)
                        true_exp_y = d.get_latent_vars()[1][:, [0]] @ d.coefficients.T
                        bias = np.mean(np.sum(np.power(true_exp_y - mean_prediction, 2), -1))

                        deviation_from_mean = np.stack(yhats, 0) - np.repeat(mean_prediction[np.newaxis, :, :],
                                                                             len(yhats), axis=0)
                        var = np.mean(np.sum(np.power(deviation_from_mean, 2), -1))
                        all_mse = np.stack([get_mse(yh[:, 0], actual_y[:, 0, :]) for yh in yhats], 0)
                        mse = float(np.mean(all_mse, 0).reshape(-1))

                        result_data.append({
                            'Model': model.name,
                            'Model_Number': model.model_id,
                            'Model_Info': json.dumps(model.get_info()),
                            'Data': d.name,
                            'Sample_Size': n,
                            'Sequence_Length': seq_len,
                            'Bias_squared': bias,
                            'Variance': var,
                            'MSE': mse,
                            'Data_Info': json.dumps(d.get_info())
                        })
        self.bias_var_results_agg = pd.DataFrame(result_data)
        self.bias_var_results_agg.to_csv(os.path.join(self.save_dir, 'bias_variance.csv'))

    def get_all_hyperparams_used(self):
        # collect all hyperparams used
        all_hyperparams = set()
        for m in self.models:
            hyperparams = list(m.get_hyperparams().keys())
            for k in hyperparams:
                all_hyperparams.add(k)
        return list(all_hyperparams)


if __name__ == '__main__':
    ex = experiment.from_path('~/2022_05_03_15_01_11')
    # ex.merge_slurm_jobs()
    # ex.compute_aggregate_scores()
    # ex.save()
