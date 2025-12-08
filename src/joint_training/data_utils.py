import os
import json
from datasets import load_dataset
from tqdm import tqdm

import pdb


entities2id = {}
relations2id = {}


def readdata(datapath):
    with open(os.path.join(datapath, 'entities.txt'), 'r') as f:
        lines = f.readlines()
        index = 0
        for entity in lines:
            entity = entity.rstrip("\n")
            entities2id[entity] = index
            index += 1 
    with open(os.path.join(datapath, 'relations.txt'), 'r') as f:
        lines = f.readlines()
        index = 0 
        for relation in lines:
            relation = relation.rstrip("\n")
            relations2id[relation] = index
            index += 1

def ParseTainData(rawpath, qatrainpath):

    data_files = os.listdir(rawpath)
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
    dataset = load_dataset('parquet', data_dir=rawpath, data_files=files, split=['train', 'test', 'validation'])
    dataset = dataset[0]
    with open(os.path.join(qatrainpath, 'subgraph.jsonl'), 'w') as fw:
        with open(os.path.join(qatrainpath, 'webqsp_train.jsonl'),  'r') as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                subgraphdict = {}
                # pdb.set_trace()
                text = line['text']
                start = text.find("Question:") + len("Question:")
                end = text.find("[/INST]")
                question = text[start:end].strip()
                # pdb.set_trace()
                if question.endswith("?"):
                    question = question[:-1]
                try:
                    index = dataset['question'].index(question)
                except:
                    print("this question is not in data")
                    continue
                datarow = dataset[index]
                subgraphdict['id'] = datarow['id']
                subgraphdict['question'] = datarow['question']
                subgraphdict['answer'] = datarow['answer']
                subgraphdict['q_entity'] = datarow['q_entity']
                subgraphdict['a_entity'] = datarow['a_entity']

                
                start = text.find("Paths:") + len("Paths:")
                end = text.find("Question")
                subgraph_list = text[start:end].strip().split('\n')
                if subgraph_list == ['']:
                    subgraphdict['subgraph'] = []
                else:
                    subgraphdict['subgraph'] = []
                    for item in subgraph_list:
                        # pdb.set_trace()
                        pathlist = item.split(' -> ')
                        length = len(pathlist)
                        s, m, e = 0, 1, 2
                        while e < length:
                            h, r, t = pathlist[s], pathlist[m], pathlist[e]
                            # pdb.set_trace()
                            hid, rid, tid = entities2id[h], relations2id[r], entities2id[t]
                            s += 2
                            e += 2
                            m += 2
                            subgraphdict['subgraph'].append([hid, rid, tid])
                answersid = []
                for answer in subgraphdict['answer']:
                    answerid = entities2id[answer]
                    answersid.append(answerid)
                subgraphdict['answer'] = answersid

                q_entitiesid = []
                for q_entity in subgraphdict['q_entity']:
                    q_entityid = entities2id[q_entity]
                    q_entitiesid.append(q_entityid)
                subgraphdict['q_entity'] = q_entitiesid

                a_entitiesid = []
                for a_entity in subgraphdict['a_entity']:
                    a_entityid = entities2id[a_entity]
                    a_entitiesid.append(a_entityid)
                subgraphdict['a_entity'] = a_entitiesid
                
                fw.write(json.dumps(subgraphdict) + "\n")
                fw.flush()





def ParseTestData(rawpath, qatrainpath):

    data_files = os.listdir(rawpath)
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
    dataset = load_dataset('parquet', data_dir=rawpath, data_files=files, split=['train', 'test', 'validation'])
    dataset = dataset[1]

    with open(os.path.join(qatrainpath, 'subgraph.jsonl'), 'w') as fw:
        with open(os.path.join(qatrainpath, 'FullRoggen3_predictions.jsonl'),  'r') as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                subgraphdict = {}
                # pdb.set_trace()
                subgraph_text = line['input']
                id = line['id']
                subgraphdict['id'] = line['id']
                index = dataset['id'].index(id)
                datarow = dataset[index]

                subgraphdict['score'] = []
                subgraphdict['question'] = datarow['question']
                subgraphdict['answer'] = datarow['answer']
                subgraphdict['q_entity'] = datarow['q_entity']
                subgraphdict['a_entity'] = datarow['a_entity']
                
                
                start = subgraph_text.find("scores:") + len("scores:")
                end = subgraph_text.find("Question:")

                subgraph_list_score = subgraph_text[start:end].strip().split('\n')

                if subgraph_list_score == ['']:
                    subgraphdict['subgraph'] = []
                else:
                    subgraphdict['subgraph'] = []
                    for item in subgraph_list_score:
                        if item == '':
                            continue
                        temp_list = []
                        # pdb.set_trace()
                        pathstr, score = item.split('\t')
                        subgraphdict['score'].append(score)
                        
                        pathlist = pathstr.split(' -> ')
                        length = len(pathlist)
                        s, m, e = 0, 1, 2
                        while e < length:
                            h, r, t = pathlist[s], pathlist[m], pathlist[e]
                            # pdb.set_trace()
                            hid, rid, tid = entities2id[h], relations2id[r], entities2id[t]
                            s += 2
                            e += 2
                            m += 2
                            temp_list.append([hid, rid, tid])
                        subgraphdict['subgraph'].append(temp_list)
                    
                answersid = []
                for answer in subgraphdict['answer']:
                    answerid = entities2id[answer]
                    answersid.append(answerid)
                subgraphdict['answer'] = answersid

                q_entitiesid = []
                for q_entity in subgraphdict['q_entity']:
                    q_entityid = entities2id[q_entity]
                    q_entitiesid.append(q_entityid)
                subgraphdict['q_entity'] = q_entitiesid

                a_entitiesid = []
                for a_entity in subgraphdict['a_entity']:
                    a_entityid = entities2id[a_entity]
                    a_entitiesid.append(a_entityid)
                subgraphdict['a_entity'] = a_entitiesid
                
                fw.write(json.dumps(subgraphdict) + "\n")
                fw.flush()



if __name__ == '__main__':
    path = 'xxxxx/data/webqsp'
    readdata(path)
    webqsp_path = 'xxxxxx/datasets/data/webqsp'
    # ParseTainData(rawpath=webqsp_path, qatrainpath=qatrainpath)
    test_path = 'xxxxx'
    ParseTestData(webqsp_path, test_path)