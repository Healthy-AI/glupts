from experiment import single_node_parallel_job
import argparse
import os
import compress_pickle

parser = argparse.ArgumentParser(description='Running a single node job on a SLURM cluster.')
parser.add_argument('--config_path', help='path to pickled configs for experiment', type=str)
parser.add_argument('--workers', help='Number of cpu cores to use', type=int)
parser.add_argument('--csv_path', help='File path for csv results.', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    path = args.config_path
    if not os.path.exists(path):
        raise Exception(f'Path does not exist! {path}')
    else:
        with open(path,'rb') as f:
            configs = compress_pickle.load(f)
        single_node_parallel_job(configs, args.workers, args.csv_path)
