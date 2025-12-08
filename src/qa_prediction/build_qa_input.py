import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable
from transformers import AutoTokenizer
# from utils import *
import datasets
import pdb


class PromptBuilder(object):
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""
    def __init__(self, prompt_path, add_rule = False, use_true = False, cot = False, explain = False, use_random = False, each_line = False, maximun_token = 4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule # 是为了区别只用LLM和使用路径辅助LLM的区别,为F就是纯用LLM进行推理
        self.use_true = use_true # use true是为了在生成的时候区别 训练和测试阶段一个使用ground truth, 一个使用预测的路径.
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line
        
    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template
    
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        # pdb.set_trace()
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res) # [[('Leo Howard', 'film.performance.actor', 'm.0cs72ly'), ('m.0cs72ly', 'film.film.starring', "Aussie & Ted's Great Adventure")]]
        return results
    
    def direct_answer(self, question_dict):
        graph = utils.build_graph(question_dict['graph'])
        entities = question_dict['q_entity']
        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
    

    def process_neg_input(self, question_dict):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']
        # pdb.set_trace()
        if not question.endswith('?'):
            question += '?'

        if self.add_rule:
            graph = utils.build_graph(question_dict['graph'])
            entities = question_dict['q_entity']
            if self.use_true:
                # pdb.set_trace()
                gt_rules = question_dict['ground_paths'] # 都是relations[('film.performance.actor', 'film.film.starring'), ('people.person.gender', 'fictional_universe.fictional_character.gender'), ...]
                neg_rules = question_dict['neg_paths']
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths'] # 这个在测试生成答案的时候用的是预测出来的路径 在训练的时候use_true为T用的是ground_paths
            # pdb.set_trace()
            if len(gt_rules) > 0:
                # pdb.set_trace()
                rule_list = []
                for item in gt_rules:
                    for p in item:
                        rule_list.append(p)
                # gt_rules = 
                # reasoning_paths = self.apply_rules(graph, gt_rules, entities)
                # pdb.set_trace()
                lists_of_gtpaths = [utils.path_to_string(p) for p in rule_list]
                # lists_of_gtpaths = gt_rules
                # context = "\n".join([utils.path_to_string(p) for p in reasoning_paths])
            else:
                lists_of_gtpaths = []
            # pdb.set_trace()

            if len(neg_rules) > 0:
                rule_list = []
                for item in neg_rules:
                    for p in item:
                        rule_list.append(p)
                # pdb.set_trace()
                # reasoning_paths = self.apply_rules(graph, neg_rules, entities)
                lists_of_negpaths = [utils.path_to_string(p) for p in rule_list]
                # lists_of_negpaths = neg_rules
                # context = "\n".join([utils.path_to_string(p) for p in reasoning_paths])
            else:
                lists_of_negpaths = []
            # pdb.set_trace()
            lists_of_paths = lists_of_gtpaths + lists_of_negpaths
            #input += self.GRAPH_CONTEXT.format(context = context)
            
        input = self.QUESTION.format(question = question) # input: question of the example
        # MCQ
        if len(question_dict['choices']) > 0:
            choices = '\n'.join(question_dict['choices'])
            input += self.CHOICES.format(choices = choices)
            if self.add_rule:
                instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ 这里一般还是一个答案 没有choice这个key
        else:
            if self.add_rule:
                instruction = self.SAQ_RULE_INSTRUCTION # 确定使用的是普通的只使用LLM的instruction 还是带有路径指导的instruction.
            else:
                instruction = self.SAQ_INSTRUCTION # 只使用LLM的instruction
        
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_rule:
            other_prompt = self.prompt_template.format(instruction = instruction, input = self.GRAPH_CONTEXT.format(context = "") + input)
            context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            # pdb.set_trace()
            input = self.GRAPH_CONTEXT.format(context = context) + input # GRAPH_CONTEXT 'Reasoning Paths:\n{context}\n\n'
            
        # pdb.set_trace()
        input = self.prompt_template.format(instruction = instruction, input = input)
            
        return input
    
    
    def process_input(self, question_dict):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']
        # pdb.set_trace()
        if not question.endswith('?'):
            question += '?'

        if self.add_rule:
            graph = utils.build_graph(question_dict['graph'])
            entities = question_dict['q_entity']
            if self.use_true:
                # pdb.set_trace()
                rules = question_dict['ground_paths'] # 都是relations[('film.performance.actor', 'film.film.starring'), ('people.person.gender', 'fictional_universe.fictional_character.gender'), ...]
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths'] # 这个在测试生成答案的时候用的是预测出来的路径 在训练的时候use_true为T用的是ground_paths
            if len(rules) > 0:
                # pdb.set_trace()
                reasoning_paths = self.apply_rules(graph, rules, entities)
                lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
                # context = "\n".join([utils.path_to_string(p) for p in reasoning_paths])
            else:
                lists_of_paths = []
            #input += self.GRAPH_CONTEXT.format(context = context)
            
        input = self.QUESTION.format(question = question) # input: question of the example
        # MCQ
        if len(question_dict['choices']) > 0:
            choices = '\n'.join(question_dict['choices'])
            input += self.CHOICES.format(choices = choices)
            if self.add_rule:
                instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ 这里一般还是一个答案 没有choice这个key
        else:
            if self.add_rule:
                instruction = self.SAQ_RULE_INSTRUCTION # 确定使用的是普通的只使用LLM的instruction 还是带有路径指导的instruction.
            else:
                instruction = self.SAQ_INSTRUCTION # 只使用LLM的instruction
        
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_rule:
            other_prompt = self.prompt_template.format(instruction = instruction, input = self.GRAPH_CONTEXT.format(context = "") + input)
            context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            
            input = self.GRAPH_CONTEXT.format(context = context) + input # GRAPH_CONTEXT 'Reasoning Paths:\n{context}\n\n'
        # pdb.set_trace()
        input = self.prompt_template.format(instruction = instruction, input = input)
            
        return input
    
    def check_prompt_length(self, prompt, list_of_paths, maximun_token):
        '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths

        else:
            # Shuffle the paths
            random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)


if __name__ == '__main__':
    prompt_path = "xxxxx"
    model_name_or_path = "xxxxx"
    model_max_length = 2048 - 200
    data_path = 'xxxxxx'
    data_name = 'cwq'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    input_builder = PromptBuilder(
        prompt_path,
        add_rule = True,
        use_true= True,
        maximun_token=model_max_length,
        tokenize=lambda x: len(tokenizer.tokenize(x)),
    )

    # pdb.set_trace()

    def formatting_prompts_func(example):
        # pdb.set_trace()
        output_label = "\n".join(example['answer'])
        # Find ground-truth paths for each Q-P pair
        graph = utils.build_graph(example["graph"])
        paths = utils.get_truth_paths(example["q_entity"], example["a_entity"], graph) # [[(h, r, t), (h, r, t)], [], ...]
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
        return {"text": output_text}

    input_dir = os.path.join(data_path, data_name)
    input_files = os.listdir(input_dir)
    files = {'train':[]}
    for file in input_files:
        if file.split('.')[-1] == 'parquet':
            if file.split('-')[0] == 'train':
                files['train'].append(file)
        else:
            pass
    # pdb.set_trace()
    input_file = os.path.join(data_path, data_name)
    train_dataset = datasets.load_dataset('parquet', data_dir=input_dir, data_files=files, split="train")

    train_data0 = train_dataset[0]
    # pdb.set_trace()
    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path))
    # with open(save_path, "w") as f:
    #     print("Processing {}...".format(data_name))
    #     print("Number of process: {}".format(N_CPUS))
    #     with mp.Pool(N_CPUS) as pool:
    #         for example in tqdm(pool.imap_unordered(formatting_prompts_func, train_dataset), total=len(train_dataset)):
    #             f.write(json.dumps(example) + "\n")
    formatting_prompts_func(train_data0)

    # train_dataset = train_dataset.map(
    #     formatting_prompts_func,
    #     remove_columns=train_dataset.column_names,
    # )

