import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from transformers import AutoTokenizer
import datasets
from qa_prediction.build_qa_input import PromptBuilder
import pdb

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

save_dir = "/workspace/longxiao/KGQA/RoG/reasoning-on-graphs-master/datasets/joint_training/qa"

prompt_path = "/workspace/longxiao/KGQA/RoG/reasoning-on-graphs-master/prompts/llama2_predict.txt"
split="train"
model_max_length = 2048 - 200
data_list = ['webqsp']
data_path = '/workspace/longxiao/KGQA/RoG/reasoning-on-graphs-master/datasets/data/' 
model_name_or_path = "/workspace/longxiao/LLM/Llama-2-7b-chat-hf"
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

def formatting_prompts_func(example):
    output_label = "\n".join(example['answer'])
     # Find ground-truth paths for each Q-P pair
    graph = build_graph(example["graph"])
    # paths = get_truth_paths(example["q_entity"], example["a_entity"], graph)
    paths = get_simple_paths(example["q_entity"], example["a_entity"], graph)
    ground_paths = set()
    for path in paths:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    example["ground_paths"] = list(ground_paths)
    # pdb.set_trace()
    output_text = (
            input_builder.process_input(example)
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

    # train_dataset = datasets.Dataset.from_dict(train_dataset[:100])
    save_path = os.path.join(save_dir, data_name, data_name + "_" + 'train_3hops' + ".jsonl")
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
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        remove_columns=train_dataset.column_names,
        num_proc=N_CPUS,
    )
    # pdb.set_trace()
    train_dataset.to_json(save_path, orient="records", lines=True)
