import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset
from datasets import Dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial
import pdb

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    """
    将test数据集和生成的路径数据集进行整合, 整合为一个统一的推测数据集qa_dataset
    features: ['id', 'question', 'answer', 'q_entity', 'a_entity', 'graph', 'choices', 'predicted_paths', 'ground_paths'],
    """
    question_to_rule = dict()
    # pdb.set_trace()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        paths_scores = data['raw_output']["norm_scores"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
            "paths_scores": paths_scores
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        sample["paths_scores"] = question_to_rule[qid]["paths_scores"]
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


def prediction(data, processed_list, model):
    # pdb.set_trace()
    # LLM = get_registed_model(args.model_name)
    # # LLM.add_args(argparser)
    # model = LLM(args)
    # model.prepare_for_inference()
    # print("Prepare pipline for inference...")

    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None
    # pdb.set_trace()
    if not question.endswith('?'):
        question += '?'
    graph = utils.build_graph(data['graph'])
    entities = data['q_entity']

    rules = data['predicted_paths']
    # rules = data['ground_paths']
    # if len(rules) > 3:
    #     rules = rules[:3]
    # if len(rules) == 0:
    #     rules = data['predicted_paths']
    # pdb.set_trace()
    paths_scores = data['paths_scores']
    # pdb.set_trace()
    if len(rules) > 0:
        reasoning_paths = utils.apply_rules(graph, rules, entities)
        # pdb.set_trace()
        dict_of_paths = utils.trans_path(reasoning_paths, rules, entities)
        # pdb.set_trace()
        # lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
        # pdb.set_trace()
        # context = "\n".join([utils.path_to_string(p) for p in reasoning_paths])
    else:
        lists_of_paths = []
    # pdb.set_trace()
    retrieve_path_with_socres = utils.entity_search_prune(graph, dict_of_paths, paths_scores, question, model)
    # pdb.set_trace()
    input = utils.constract_reasoning_prompt(retrieve_path_with_socres, question)
    # pdb.set_trace()
    # pdb.set_trace()
    prediction = model.generate_sentence(input)
    if prediction is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input,
    }
    return result


def main(args):
    input_dir = os.path.join(args.data_path, args.d)
    # output_dir = os.path.join(args.output_path, args.d)
    data_files = os.listdir(input_dir)
    hit0_list = ['WebQTest-1544', 'WebQTest-663', 'WebQTest-1248', 'WebQTest-1047', 'WebQTest-614', 'WebQTest-1416', 'WebQTest-1149', 'WebQTest-1448', 'WebQTest-822', 'WebQTest-1309', 'WebQTest-76', 'WebQTest-1088', 'WebQTest-1217', 'WebQTest-612', 'WebQTest-257', 'WebQTest-792', 'WebQTest-1042', 'WebQTest-1799', 'WebQTest-1932', 'WebQTest-1987', 'WebQTest-1201', 'WebQTest-1731', 'WebQTest-308', 'WebQTest-1452', 'WebQTest-1119', 'WebQTest-768', 'WebQTest-134', 'WebQTest-1163', 'WebQTest-155', 'WebQTest-2001', 'WebQTest-1317', 'WebQTest-905', 'WebQTest-1402', 'WebQTest-1789', 'WebQTest-639', 'WebQTest-406', 'WebQTest-1436', 'WebQTest-987', 'WebQTest-1461', 'WebQTest-1740', 'WebQTest-532', 'WebQTest-1072', 'WebQTest-1287', 'WebQTest-1635', 'WebQTest-1179', 'WebQTest-993', 'WebQTest-421', 'WebQTest-182', 'WebQTest-1812', 'WebQTest-1133', 'WebQTest-3', 'WebQTest-2027', 'WebQTest-1776', 'WebQTest-1312', 'WebQTest-1421', 'WebQTest-1504', 'WebQTest-564', 'WebQTest-936', 'WebQTest-2005', 'WebQTest-12', 'WebQTest-1715', 'WebQTest-592', 'WebQTest-432', 'WebQTest-1586', 'WebQTest-1619', 'WebQTest-1052', 'WebQTest-1378', 'WebQTest-1534', 'WebQTest-1555', 'WebQTest-1994', 'WebQTest-711', 'WebQTest-708', 'WebQTest-1051', 'WebQTest-953', 'WebQTest-695', 'WebQTest-32', 'WebQTest-800', 'WebQTest-932', 'WebQTest-1926', 'WebQTest-819', 'WebQTest-58', 'WebQTest-102', 'WebQTest-1446', 'WebQTest-241', 'WebQTest-106', 'WebQTest-1348', 'WebQTest-609', 'WebQTest-1763', 'WebQTest-1050', 'WebQTest-386', 'WebQTest-401', 'WebQTest-497', 'WebQTest-1170', 'WebQTest-1169', 'WebQTest-1332', 'WebQTest-1533', 'WebQTest-561', 'WebQTest-1103', 'WebQTest-59', 'WebQTest-1410', 'WebQTest-1231', 'WebQTest-769', 'WebQTest-375', 'WebQTest-115', 'WebQTest-209', 'WebQTest-141', 'WebQTest-631', 'WebQTest-1607', 'WebQTest-1744', 'WebQTest-1975', 'WebQTest-1265', 'WebQTest-488', 'WebQTest-846', 'WebQTest-28', 'WebQTest-417', 'WebQTest-1143', 'WebQTest-759', 'WebQTest-482', 'WebQTest-1472', 'WebQTest-1620', 'WebQTest-1254', 'WebQTest-289', 'WebQTest-1719', 'WebQTest-2031', 'WebQTest-1969', 'WebQTest-1919', 'WebQTest-489', 'WebQTest-523', 'WebQTest-1089', 'WebQTest-1394', 'WebQTest-1649', 'WebQTest-1800', 'WebQTest-418', 'WebQTest-456', 'WebQTest-1599', 'WebQTest-187', 'WebQTest-1657', 'WebQTest-670', 'WebQTest-707', 'WebQTest-758', 'WebQTest-277', 'WebQTest-991', 'WebQTest-1234', 'WebQTest-1779', 'WebQTest-39', 'WebQTest-1996', 'WebQTest-1388', 'WebQTest-86', 'WebQTest-866', 'WebQTest-1803', 'WebQTest-107', 'WebQTest-876', 'WebQTest-955', 'WebQTest-1209', 'WebQTest-1849', 'WebQTest-1294', 'WebQTest-1957', 'WebQTest-504', 'WebQTest-780', 'WebQTest-1298', 'WebQTest-1654', 'WebQTest-170', 'WebQTest-1339', 'WebQTest-119', 'WebQTest-1329', 'WebQTest-882', 'WebQTest-1494', 'WebQTest-385', 'WebQTest-359', 'WebQTest-1177', 'WebQTest-713', 'WebQTest-1443', 'WebQTest-880', 'WebQTest-683', 'WebQTest-1271', 'WebQTest-1546', 'WebQTest-1357', 'WebQTest-278', 'WebQTest-1453', 'WebQTest-51', 'WebQTest-521', 'WebQTest-1153', 'WebQTest-1868', 'WebQTest-1539', 'WebQTest-1166', 'WebQTest-1864', 'WebQTest-1272', 'WebQTest-1705', 'WebQTest-1738', 'WebQTest-1964', 'WebQTest-337', 'WebQTest-1350', 'WebQTest-1499', 'WebQTest-78', 'WebQTest-1565', 'WebQTest-1537', 'WebQTest-729', 'WebQTest-740', 'WebQTest-1570', 'WebQTest-1462', 'WebQTest-2000', 'WebQTest-997', 'WebQTest-1418', 'WebQTest-31', 'WebQTest-1015', 'WebQTest-1626', 'WebQTest-619', 'WebQTest-578', 'WebQTest-771', 'WebQTest-718']
    list1 = ['WebQTest-277', 'WebQTest-1088','WebQTest-1738','WebQTest-1739','WebQTest-1760','WebQTest-417','WebQTest-1868','WebQTest-1254','WebQTest-1880','WebQTest-504','WebQTest-556','WebQTest-1996','WebQTest-664','WebQTest-2005','WebQTest-707','WebQTest-729','WebQTest-114','WebQTest-1544','WebQTest-865','WebQTest-936', 'WebQTest-991']
    without_path_list = ['WebQTest-1051', 'WebQTest-1091', 'WebQTest-1131', 'WebQTest-1133', 'WebQTest-1144', 'WebQTest-1149', 'WebQTest-1153', 'WebQTest-1163', 'WebQTest-1166', 'WebQTest-1170', 'WebQTest-1234', 'WebQTest-1272', 'WebQTest-1294', 'WebQTest-1332', 'WebQTest-1348', 'WebQTest-1399', 'WebQTest-1408', 'WebQTest-1418', 'WebQTest-1446', 'WebQTest-1472', 'WebQTest-1533', 'WebQTest-1620', 'WebQTest-1635', 'WebQTest-1646', 'WebQTest-1649', 'WebQTest-1695', 'WebQTest-1779', 'WebQTest-1791', 'WebQTest-1800', 'WebQTest-1888', 'WebQTest-1902', 'WebQTest-1925', 'WebQTest-1926', 'WebQTest-1957', 'WebQTest-1979', 'WebQTest-3', 'WebQTest-31', 'WebQTest-32', 'WebQTest-38', 'WebQTest-46', 'WebQTest-59', 'WebQTest-106', 'WebQTest-155', 'WebQTest-172', 'WebQTest-245', 'WebQTest-273', 'WebQTest-289', 'WebQTest-317', 'WebQTest-337', 'WebQTest-379', 'WebQTest-385', 'WebQTest-386', 'WebQTest-390', 'WebQTest-482', 'WebQTest-521', 'WebQTest-568', 'WebQTest-592', 'WebQTest-614', 'WebQTest-631', 'WebQTest-639', 'WebQTest-663', 'WebQTest-718', 'WebQTest-771', 'WebQTest-846', 'WebQTest-882', 'WebQTest-905', 'WebQTest-932', 'WebQTest-948', 'WebQTest-954', 'WebQTest-963', 'WebQTest-997']
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
    rule_postfix = "no_rule"
    # Load dataset
    dataset = load_dataset('parquet', data_dir=input_dir, data_files=files, split=['train', 'test', 'validation'])
    dataset = dataset[1]
    # pdb.set_trace()
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"
    # pdb.set_trace()
    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
        
    print("Load dataset from finished")
    # pdb.set_trace()
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, "seperate2"
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    LLM = get_registed_model(args.model_name)
    if LLM is not None:
        model = LLM(args)
        # input_builder = PromptBuilder(
        #     args.prompt_path,
        #     args.add_rule,
        #     use_true=args.use_true,
        #     cot=args.cot,
        #     explain=args.explain,
        #     use_random=args.use_random,
        #     each_line=args.each_line,
        #     maximun_token=model.maximun_token,
        #     tokenize=model.tokenize,
        # )
        print("Prepare pipline for inference...")
        # pdb.set_trace()
        model.prepare_for_inference()
    # else:
    #     model = None
    #     # Directly return last entity as answer
    #     input_builder = PromptBuilder(
    #         args.prompt_path, args.add_rule, use_true=args.use_true
    #     )
    # pdb.set_trace()
    # Save args file
    with open(os.path.join(output_dir, "my_args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"FullRoggen33_beam_predictions.jsonl")
    # pdb.set_trace()
    fout, processed_list = get_output_file(output_file, force=args.force)
    # pdb.set_trace()
    dataset = Dataset.from_dict(dataset[1100:])
    if args.n > 1:
        with Pool(args.n) as p:
            for res in list(tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            )):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        # pdb.set_trace()
        for data in tqdm(dataset):
            print(data['id']) #1248 1072
            # if data['id'] != 'WebQTest-1042':
            #     continue
            # if data['id'] not in hit0_list:
            #     continue
            if data['id'] in without_path_list:
                continue
            res = prediction(data, processed_list, model)
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    # pdb.set_trace()
    fout.close()

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path", type=str, default="xxxxx"
    )
    argparser.add_argument("--d", "-d", type=str, default="webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="xxxxxx")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="xxxxxxx",
    )
    argparser.add_argument("--add_rule", action="store_true") # T
    argparser.add_argument("--use_true", action="store_true") # F
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="xxxxxxx",
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--model_path", type=str, default='xxxxxx')

    argparser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
    argparser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], type=str, default='fp16')
    # pdb.set_trace()
    args, _ = argparser.parse_known_args()

    # if args.model_name != "no-llm":
    #     LLM = get_registed_model(args.model_name)
    #     LLM.add_args(argparser)
    # else:
    #     LLM = None
    # args = argparser.parse_args()
    # pdb.set_trace()
    main(args)




    