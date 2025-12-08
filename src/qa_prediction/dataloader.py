import os, sys
import numpy as np
from tqdm import tqdm
import json
import pdb
from transformers import AutoTokenizer, LlamaForCausalLM


class BasicDataLoader(object):
    def __init__(self, config, tokenize, data_type="train"):
        self._parse_args(config)
        self._load_file(config, data_type)
        self._load_data()
        self.tokenize = tokenize

    def _parse_args(self, config):
        """
        Builds necessary dictionaries and stores arguments.
        """
        self.data_name = config['name']
        self.entity2id = {}
        self.relation2id = {}
        datapath = config['rawdatapath']

        with open(os.path.join(datapath, 'entities.txt'), 'r') as f:
            lines = f.readlines()
            index = 0
            for entity in lines:
                entity = entity.rstrip("\n")
                self.entity2id[entity] = index
                index += 1 
        with open(os.path.join(datapath, 'relations.txt'), 'r') as f:
            lines = f.readlines()
            index = 0 
            for relation in lines:
                relation = relation.rstrip("\n")
                self.relation2id[relation] = index
                index += 1

        if 'use_inverse_relation' in config:
            self.use_inverse_relation = config['use_inverse_relation']
        else:
            self.use_inverse_relation = False
        if 'use_self_loop' in config:
            self.use_self_loop = config['use_self_loop']
        else:
            self.use_self_loop = False

        #self.num_step = config['num_step']
        self.max_local_entity = 0
        self.max_relevant_doc = 0
        self.max_facts = 0

        self.id2entity = {i: entity for entity, i in self.entity2id.items()} # 这个等会需要做初始化的

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(self.relation2id)
        else:
            self.num_kb_relation = len(self.relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        print("Entity: {}, Relation in KB: {}, Relation in use: {} ".format(len(self.entity2id),
                                                                            len(self.relation2id),
                                                                            self.num_kb_relation))

    def _load_file(self, config, data_type="train"):

        """
        Loads lines (questions + KG subgraphs) from json files.
        """
        
        subgraph_file = config['subgraph_folder'] + data_type + ".jsonl"
        self.data_file = subgraph_file
        print('loading data from', subgraph_file)
        self.data_type = data_type
        self.data = [] # [{}, ...{}]
        skip_index = set()
        # index = 0

        with open(subgraph_file) as f_in:
            for line in tqdm(f_in):
                line = json.loads(line)
                index = line['id']
                # pdb.set_trace()
                if len(line['subgraph']) == 0:
                    skip_index.add(index)
                    continue
                self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']))
                index += 1

        print("skip", skip_index)
        print('max_facts: ', self.max_facts, 'num_data: ', len(self.data))
        # pdb.set_trace()
        self.num_data = len(self.data) # 这里存的是所有具有subgraph的数据
        self.batches = np.arange(self.num_data)


    def _load_data(self):
        """
        Creates mappings between global entity ids and local entity ids that are used during GNN updates.
        """
        print('converting global to local entity index ...')
        # pdb.set_trace()
        self.global2local_entity_maps = self._build_global2local_entity_maps() # list [{}, {}, ..., {}] 每一个都是每个数据集里面的实体id转化表 由全局的id转化为局部的id {23410:0, ...}
        # pdb.set_trace()
        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity # max_facts:18992 + 2000 = 20992

        self.question_id = [] # ['WebQTrn-9', 'WebQTrn-11', 'WebQTrn-15', ...]
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int) # 装的是global的id 每个问题子图最大2000个实体 装了在这个子图中对应实体的真实id
        self.kb_adj_mats = np.empty(self.num_data, dtype=object) # 存的是局部的结构 (array([1872,375,128, ...,]), array([ 54, 4,133,]), array([ 144, 1, 852,...,]))
        self.q_adj_mats = np.empty(self.num_data, dtype=object)
        self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int) #  (249, 20992)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float) # 这个是个one hot向量

        self._prepare_data()





    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        
        max_count = 0
        for line in self.data:
            word_list = line["question"].split(' ')
            max_count = max(max_count, len(word_list))

        
        self.build_rel_words(self.tokenize)

        # pdb.set_trace()

        self.max_query_word = max_count
        #self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        #self.query_texts2 = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)

        #build tokenizers
        if self.tokenize == 'lstm':
            self.num_word = len(self.word2id)
            self.tokenizer = LSTMTokenizer(self.word2id, self.max_query_word)
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)
        else:
            if self.tokenize == 'bert':
                tokenizer_name = 'bert-base-uncased'    
            elif self.tokenize  == 'roberta':
                tokenizer_name = 'roberta-base'
            elif self.tokenize  == 'sbert':
                tokenizer_name = 'sentence-transformers/all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a/'
            elif self.tokenize == 'sbert2':
                tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
            elif self.tokenize  == 't5':
                tokenizer_name = 't5-small'
            elif self.tokenize == 'simcse':
                tokenizer_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
            elif self.tokenize  == 't5':
                tokenizer_name = 't5-small'
            elif self.tokenize  == 'relbert':
                tokenizer_name = 'pretrained_lms/sr-simbert/'

            self.max_query_word = max_count + 2 #for cls token and sep
            #self.tokenizer = AutoTokenizer(self.max_query_word)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # pdb.set_trace()
            self.num_word = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) #self.tokenizer.q_tokenizer.encode("[UNK]")[0]
            
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int) # 把所有的query填充为0 pad的id

        next_id = 0
        num_query_entity = {}
        for sample in tqdm(self.data): # self.data 装的是所有的训练 测试数据集 dict
            self.question_id.append(sample["id"])
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id] # g2l是global2local的id转化
            #print(g2l)
            if len(g2l) == 0:
                #print(next_id)
                continue
            # build connection between question and entities in it
            tp_set = set()
            seed_list = []
            key_ent = 'entities_cid' if 'entities_cid' in sample else 'entities'
            for j, entity in enumerate(sample[key_ent]):
                # if entity['text'] not in self.entity2id:
                #     continue
                try:
                    if isinstance(entity, dict) and  'text' in entity:
                        global_entity = self.entity2id[entity['text']]
                    else:
                        global_entity = self.entity2id[entity]
                    global_entity = self.entity2id[entity['text']]
                except:
                    global_entity = entity #self.entity2id[entity['text']]

                if global_entity not in g2l:
                    continue
                local_ent = g2l[global_entity]
                self.query_entities[next_id, local_ent] = 1.0
                seed_list.append(local_ent)
                tp_set.add(local_ent)
            
            # pdb.set_trace()
            self.seed_list[next_id] = seed_list
            num_query_entity[next_id] = len(tp_set)
            for global_entity, local_entity in g2l.items():
                if self.data_name != 'cwq':

                    if local_entity not in tp_set:  # skip entities in question
                    #print(global_entity)
                    #print(local_entity)
                        self.candidate_entities[next_id, local_entity] = global_entity # 装的是global的id
                elif self.data_name == 'cwq':
                    self.candidate_entities[next_id, local_entity] = global_entity
                # if local_entity != 0:  # skip question node
                #     self.candidate_entities[next_id, local_entity] = global_entity
            # pdb.set_trace()
            # relations in local KB
            head_list = []
            rel_list = []
            tail_list = []
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                try:
                    if isinstance(sbj, dict) and  'text' in sbj:
                        head = g2l[self.entity2id[sbj['text']]]
                        rel = self.relation2id[rel['text']]
                        tail = g2l[self.entity2id[obj['text']]]
                    else:
                        head = g2l[self.entity2id[sbj]]
                        rel = self.relation2id[rel]
                        tail = g2l[self.entity2id[obj]]
                except:
                    head = g2l[sbj]
                    try:
                        rel = int(rel)
                    except:
                        rel = self.relation2id[rel]
                    tail = g2l[obj]
                head_list.append(head)
                rel_list.append(rel)
                tail_list.append(tail)
                self.kb_fact_rels[next_id, i] = rel
                if self.use_inverse_relation:
                    head_list.append(tail)
                    rel_list.append(rel + len(self.relation2id))
                    tail_list.append(head)
                    self.kb_fact_rels[next_id, i] = rel + len(self.relation2id)
                
            if len(tp_set) > 0:
                for local_ent in tp_set:
                    self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
            else:
                for index in range(len(g2l)):
                    self.seed_distribution[next_id, index] = 1.0 / len(g2l)
            try:
                assert np.sum(self.seed_distribution[next_id]) > 0.0
            except:
                print(next_id, len(tp_set))
                exit(-1)

            #tokenize question
            if self.tokenize == 'lstm':
                self.query_texts[next_id] = self.tokenizer.tokenize(sample['question'])
            else:
                tokens =  self.tokenizer.encode_plus(text=sample['question'], max_length=self.max_query_word, \
                    pad_to_max_length=True, return_attention_mask = False, truncation=True)
                self.query_texts[next_id] = np.array(tokens['input_ids'])


            # construct distribution for answers
            answer_list = []
            if 'answers_cid' in sample:
                for answer in sample['answers_cid']:
                    #keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                    answer_ent = answer
                    answer_list.append(answer_ent)
                    if answer_ent in g2l:
                        self.answer_dists[next_id, g2l[answer_ent]] = 1.0
            else:
                for answer in sample['answers']:
                    keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                    answer_ent = self.entity2id[answer[keyword]]
                    answer_list.append(answer_ent)
                    if answer_ent in g2l:
                        self.answer_dists[next_id, g2l[answer_ent]] = 1.0
            self.answer_lists[next_id] = answer_list

            if not self.data_eff:
                self.kb_adj_mats[next_id] = (np.array(head_list, dtype=int),
                                         np.array(rel_list, dtype=int),
                                         np.array(tail_list, dtype=int))

            next_id += 1
        num_no_query_ent = 0
        num_one_query_ent = 0
        num_multiple_ent = 0
        for i in range(next_id):
            ct = num_query_entity[i]
            if ct == 1:
                num_one_query_ent += 1
            elif ct == 0:
                num_no_query_ent += 1
            else:
                num_multiple_ent += 1
        print("{} cases in total, {} cases without query entity, {} cases with single query entity,"
              " {} cases with multiple query entities".format(next_id, num_no_query_ent,
                                                              num_one_query_ent, num_multiple_ent))



    def build_rel_words(self, tokenize):
        """ 
        Tokenizes relation surface forms.
        """
        if tokenize == 'llama':
            tokenizer_path= "xxxxxx"
        elif tokenize == 'roberta':
            tokenizer_path = 'roberta-base'
        elif tokenize == 'sbert':
            tokenizer_path = 'sentence-transformers/all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a/'
        elif tokenize == 'sbert2':
            tokenizer_path = 'sentence-transformers/all-mpnet-base-v2'
        elif tokenize == 'simcse':
            tokenizer_path = 'princeton-nlp/sup-simcse-bert-base-uncased'
        elif tokenize  == 'relbert':
            tokenizer_path = 'pretrained_lms/sr-simbert/'
        max_rel_words = 0
        rel_words = []
        # pdb.set_trace()
        if 'metaqa' in self.data_file:
            for rel in self.relation2id:
                words = rel.split('_')
                max_rel_words = max(len(words), max_rel_words)
                rel_words.append(words)
            #print(rel_words)
        else:
            for rel in self.relation2id:
                rel = rel.strip()
                fields = rel.split('.')
                try:
                    words = fields[-2].split('_') + fields[-1].split('_')
                    max_rel_words = max(len(words), max_rel_words)
                    rel_words.append(words)
                    #print(rel, words)
                except:
                    words = ['UNK']
                    rel_words.append(words)
                    pass
        # pdb.set_trace()    
        self.max_rel_words = max_rel_words

            
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        pad_val = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int) # (6104, 12)
        self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int)
        # pdb.set_trace()
        for rel_id,words in enumerate(rel_words):
            tokens =  tokenizer.encode_plus(text=' '.join(words), max_length=self.max_rel_words, \
                pad_to_max_length=True, return_attention_mask = False, truncation=True)
            tokens_inv =  tokenizer.encode_plus(text=' '.join(words[::-1]), max_length=self.max_rel_words, \
                pad_to_max_length=True, return_attention_mask = False, truncation=True)
            self.rel_texts[rel_id] = np.array(tokens['input_ids'])
            self.rel_texts_inv[rel_id] = np.array(tokens_inv['input_ids'])
        # pdb.set_trace()

        assert len(rel_words) == len(self.relation2id)
        #print(self.rel_texts, self.max_rel_words)


    def build_new_rel_words(self, tokenize):
        """ 
        Tokenizes relation surface forms.
        """

        if tokenize == 'llama':
            tokenizer_path= "xxxxxxx"
        elif tokenize == 'roberta':
            tokenizer_path = 'roberta-base'
        elif tokenize == 'sbert':
            tokenizer_path = 'sentence-transformers/all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a/'
        elif tokenize == 'sbert2':
            tokenizer_path = 'sentence-transformers/all-mpnet-base-v2'
        elif tokenize == 'simcse':
            tokenizer_path = 'princeton-nlp/sup-simcse-bert-base-uncased'
        elif tokenize  == 'relbert':
            tokenizer_path = 'pretrained_lms/sr-simbert/'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        if 'webqsp' in self.data_file:
            self.max_rel_words = 12
        if 'cwq' in self.data_file:
            self.max_rel_words = 10

        pad_val = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int) # (6104, 12)
        self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int)
        # pdb.set_trace()
        for (words, rel_id) in self.relation2id.items():
            tokens =  tokenizer.encode_plus(text=words, max_length=self.max_rel_words, \
                pad_to_max_length=True, return_attention_mask = False, truncation=True)
            tokens_inv =  tokenizer.encode_plus(text=' '.join(words[::-1]), max_length=self.max_rel_words, \
                pad_to_max_length=True, return_attention_mask = False, truncation=True)
            self.rel_texts[rel_id] = np.array(tokens['input_ids'])
            self.rel_texts_inv[rel_id] = np.array(tokens_inv['input_ids'])
        # pdb.set_trace()

        assert len(rel_words) == len(self.relation2id)
        #print(self.rel_texts, self.max_rel_words)




    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        # pdb.set_trace()
        for sample in tqdm(self.data):
            g2l = dict()
            answer = sample['answer']
            q_entity = sample['q_entity']
            subgraph = sample['subgraph']
            self._add_entity_to_map(answer, g2l)
            self._add_entity_to_map(q_entity, g2l)
            #self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(subgraph, g2l)
            # pdb.set_trace()
            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        print('max local entity: ', self.max_local_entity)
        return global2local_entity_maps

    @staticmethod
    def _add_entity_to_map(entities, g2l):
        #print(entities)
        #print(entity2id)
        if isinstance(entities[0], list):
            for item in entities:
                h, r, t = item
                if h not in g2l:
                    g2l[h] = len(g2l)
                if t not in g2l:
                    g2l[t] = len(g2l)
        else:
            for entity_global_id in entities:
                if entity_global_id not in g2l:
                    g2l[entity_global_id] = len(g2l)



class SingleDataLoader(BasicDataLoader):
    """
    Single Dataloader creates training/eval batches during KGQA.
    """
    def __init__(self, config, tokenize, data_type="train"):
        super(SingleDataLoader, self).__init__(config, tokenize, data_type)
        
    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        # pdb.set_trace()
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        # true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        # self.sample_ids = sample_ids
        # self.true_sample_ids = ori_sample_ids
        # self.batch_ids = true_batch_id
        question_input = self.query_texts[sample_ids]
        
        kb_adj_mats = self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)
        pdb.set_trace()
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mats, \
                   question_input, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               kb_adj_mats, \
               question_input, \
               self.answer_dists[sample_ids]


if __name__ == "__main__":
    config = {}