import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import os
import json
from datasets import load_dataset
import multiprocessing as mp
import utils
from tqdm import tqdm
from functools import partial
import pdb


def build_data(args):
    '''
    Extract the paths between question and answer entities from the dataset.
    '''
    
    input_dir = os.path.join(args.data_dir, args.d)
    output_dir = os.path.join(args.output_path, args.d)
    data_files = os.listdir(input_dir)
    files = {'train':[], 'test':[], 'validation': []}
    for file in data_files:
        if file.split('.')[-1] == 'parquet':
            if file.split('-')[0] == 'train':
                files['train'].append(file)
            if file.split('-')[0] == 'test':
                files['test'].append(file)
            if file.split('-')[0] == 'validation':
                files['validation'].append(file)
        else:
            pass
            
    # pdb.set_trace()
    

    print("Save results to: ", output_dir)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    # Load dataset
    dataset = load_dataset('parquet', data_dir=input_dir, data_files=files, split=['train', 'test', 'validation'])
    
    dataset = dataset[0]
    # for data in dataset:
    #     if len(data['q_entity']) == 0:
    #         print('q_entity is not 1', data['id'])
    #     if len(data['a_entity']) == 0:
    #         print('a_entity is not 1', data['id'])
    # pdb.set_trace()
    # pdb.set_trace()

    with open(os.path.join(output_dir, args.save_name), 'w') as fout:
        with mp.Pool(args.n) as pool:
            for res in tqdm(pool.imap_unordered(partial(process_data, remove_duplicate=args.remove_duplicate), dataset), total=len(dataset)):
                for r in res:
                    fout.write(json.dumps(r) + '\n')
    # for data in dataset:
    #     result = process_data(data)


def process_data(data, remove_duplicate=True):
    question = data['question']
    graph  =  utils.build_graph(data['graph'])
    paths = utils.get_truth_paths(data['q_entity'], data['a_entity'], graph)
    # paths = utils.get_simple_paths(data['q_entity'], data['a_entity'], graph)
    result = []
    # Split each Q-P pair into a single data
    rel_paths = []
    for path in paths:
        rel_path = [p[1] for p in path] # extract relation path
        if remove_duplicate:
            if tuple(rel_path) in rel_paths:
                continue
        rel_paths.append(tuple(rel_path))
    # pdb.set_trace()
    results = utils.apply_rules_select(graph, rel_paths, data['q_entity'], data['a_entity'])

    for rel_path in results:
        result.append({"question": question, "path": rel_path}) 
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="parquet")
    parser.add_argument("--d", type=str, default='cwq')
    parser.add_argument('--data_dir', type=str, default='xxxx')
    # parser.add_argument('data_files', default={"test":""})
    parser.add_argument('--split', type=str, default=['train', 'test', 'validation'])
    parser.add_argument("--output_path", type=str, default="xxxx")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument('--n', '-n', type=int, default=1)
    parser.add_argument('--remove_duplicate', action='store_true', default=True)
    args = parser.parse_args()
    
    if args.save_name == "":
        args.save_name = args.d + "_" + args.split[0] + "_" + '3hops_filter' + ".jsonl"
    
    build_data(args)
