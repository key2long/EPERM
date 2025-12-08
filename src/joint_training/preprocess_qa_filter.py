import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
import utils
from transformers import AutoTokenizer
import datasets
from qa_prediction.build_qa_input import PromptBuilder
import pdb
from datasets import Dataset

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

save_dir = "xxxxx"

prompt_path = "xxxxxx"
split="train"
model_max_length = 2048 - 200
data_list = ['cwq']
data_path = 'xxxxxxx' 
model_name_or_path = "xxxxxx"
prompter = InstructFormater(prompt_path)

tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )


# Load prompt template
input_builder = PromptBuilder(
        prompt_path,
        add_rule = True,
        use_true= True,
        maximun_token=model_max_length,
        tokenize=lambda x: len(tokenizer.tokenize(x)),
    )


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    """
    将test数据集和生成的路径数据集进行整合, 整合为一个统一的推测数据集qa_dataset
    features: ['id', 'question', 'answer', 'q_entity', 'a_entity', 'graph', 'choices', 'predicted_paths', 'ground_paths'],
    """
    question_to_rule = dict()
    rule_qid = []
    # pdb.set_trace()
    for data in rule_dataset:
        qid = data["id"]
        neg_paths = []
        rule_qid.append(qid)
        gt_paths = data["ground_paths"]
        pt_paths = data["prediction"]
        for pt_path in pt_paths:
            if pt_path in gt_paths:
                continue
            else:
                neg_paths.append(pt_path)
        
        question_to_rule[qid] = {
            "neg_paths": neg_paths,
            "ground_paths": gt_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        if qid not in rule_qid:
            sample["neg_paths"] = []
            sample["ground_paths"] = []
        else:
            sample["neg_paths"] = []
            sample["ground_paths"] = []
            sample["neg_paths"] = question_to_rule[qid]["neg_paths"]
            sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.
    # pdb.set_trace()
    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    # pdb.set_trace()
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    # pdb.set_trace()
    return qa_dataset


def formatting_prompts_func(example):
    output_label = "\n".join(example['answer'])
     # Find ground-truth paths for each Q-P pair
    graph = build_graph(example["graph"])
    neg_paths = example['neg_paths']
    paths = get_truth_paths(example["q_entity"], example["a_entity"], graph)
    # paths = get_simple_paths(example["q_entity"], example["a_entity"], graph)
    ground_paths = set()
    for path in paths:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path

    # pdb.set_trace()
    pos_results = utils.apply_rules(graph, list(ground_paths), example['q_entity'])
    neg_results = utils.apply_rules(graph, list(neg_paths), example['q_entity'])

    # pdb.set_trace()
    example["ground_paths"] = pos_results
    example["neg_paths"] = neg_results
    # pdb.set_trace()
    output_text = (
            input_builder.process_neg_input(example)
            + " "
            + output_label + tokenizer.eos_token
        )
    # pdb.set_trace()
    return {"text": output_text}


for data_name in data_list:

    input_dir = os.path.join(data_path, data_name)
    input_files = os.listdir(input_dir)
    files = {'train':[], 'test':[], 'validation': []}
    for file in input_files:
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
    # input_file = os.path.join(data_path, data_name)
    dataset = datasets.load_dataset('parquet', data_dir=input_dir, data_files=files, split=['train', 'test', 'validation'])
    train_dataset = dataset[0]
    rule_dataset = utils.load_jsonl('xxxxx')

    train_dataset = merge_rule_result(train_dataset, rule_dataset)
    # pdb.set_trace()
    # train_dataset = datasets.Dataset.from_dict(train_dataset[:100])
    save_path = os.path.join(save_dir, data_name, data_name + "_" + 'train_with_neg' + ".jsonl")
    # pdb.set_trace()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # with open(save_path, "w") as f:
    #     print("Processing {}...".format(data_name))
    #     print("Number of process: {}".format(N_CPUS))
    #     with mp.Pool(N_CPUS) as pool:
    #         for example in tqdm(pool.imap_unordered(formatting_prompts_func, train_dataset), total=len(train_dataset)):
    #             f.write(json.dumps(example) + "\n")
    # pdb.set_trace()
    # for data in train_dataset:
    #     formatting_prompts_func(data)
    # train_dataset = Dataset.from_dict(train_dataset[:10])
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        remove_columns=train_dataset.column_names,
        num_proc=N_CPUS,
    )
    # pdb.set_trace()
    train_dataset.to_json(save_path, orient="records", lines=True)
