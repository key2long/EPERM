import networkx as nx
from collections import deque, defaultdict
import walker
import pdb
import re
import random

import sys
# sys.path.append('/workspace/longxiao/KGQA/RoG/reasoning-on-graphs-master/src/utils/')
# pdb.set_trace()
from .prompt_list import *
# from prompt_list import score_entity_candidates_prompt, reasoning_prompt

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

# 定义一个函数来进行宽度优先搜索
def bfs_with_rule(graph, start_node, target_rule, max_p = 10):
    result_paths = []
    queue = deque([(start_node, [])])  # 使用队列存储待探索节点和对应路径
    while queue:
        current_node, current_path = queue.popleft()

        # 如果当前路径符合规则，将其添加到结果列表中
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            # if len(result_paths) >= max_p:
            #     break
            
        # 如果当前路径长度小于规则长度，继续探索
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                # 剪枝：如果当前边类型与规则中的对应位置不匹配，不继续探索该路径
                rel = graph[current_node][neighbor]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(target_rule):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))
    
    return result_paths



def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities. [[(h, r, t), (h, r, t)], [], ...]
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths


def apply_rules_select(graph, rules, srouce_entity: list, target_entity: list, topk=3):
    results = []
    rule_score = {}
    # pdb.set_trace()
    for rule in rules:
        hit = 0.0
        for s_entity in srouce_entity:
            for t_entity in target_entity:
                # pdb.set_trace()
                res = bfs_with_rule(graph, s_entity, rule)
                retrive_len = len(res)
                if retrive_len == 0:
                    # print(res)
                    continue
                for retrive_path in res:
                    try:
                        retrive_result = retrive_path[-1][-1]
                    except:
                        continue
                    if retrive_result == t_entity:
                        hit += 1
                try:
                    hit_rate = hit / retrive_len
                except:
                    pass

                # pdb.set_trace()
        rule_score[rule] = hit_rate
    # pdb.set_trace()
        # if hit_rate >= select_rate:
        #     results.append(rule) # [[('Leo Howard', 'film.performance.actor', 'm.0cs72ly'), ('m.0cs72ly', 'film.film.starring', "Aussie & Ted's Great Adventure")]]
    sorted_rule = sorted(rule_score.items(), key=lambda x:x[1], reverse=True)
    hit1_path_list = []

    for path_with_score in sorted_rule:
        if path_with_score[-1] == 1.0:
            hit1_path_list.append(path_with_score[0])

    # pdb.set_trace()
    if len(sorted_rule) >= topk:
        if len(hit1_path_list) > topk:
            results = hit1_path_list
        else:
            for rule_score in sorted_rule[:topk]:
                results.append(rule_score[0])
    else:
        for rule_score in sorted_rule:
            results.append(rule_score[0])
    # pdb.set_trace()
    return results


def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    '''
    Get all simple paths connecting question and answer entities within given hop
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append([(e[0], graph[e[0]][e[1]]['relation'], e[1]) for e in p])
    return result_paths


def get_negative_paths(q_entity: list, a_entity: list, graph: nx.Graph, n_neg: int, hop=2) -> list:
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    end_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    for t in a_entity:
        if t in graph:
            end_nodes.append(node_idx.index(t))
    paths = walker.random_walks(graph, n_walks=n_neg, walk_len=hop, start_nodes=start_nodes, verbose=False)
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        # remove paths that end with answer entity
        if p[-1] in end_nodes:
            continue
        for i in range(len(p)-1):
            u = node_idx[p[i]]
            v = node_idx[p[i+1]]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths

def get_random_paths(q_entity: list, graph: nx.Graph, n=3, hop=2):
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    paths = walker.random_walks(graph, n_walks=n, walk_len=hop, start_nodes=start_nodes, verbose=False)
    # Add relation to paths
    result_paths = []
    rules = []
    for p in paths:
        tmp = []
        tmp_rule = []
        for i in range(len(p)-1):
            u = node_idx[p[i]]
            v = node_idx[p[i+1]]
            tmp.append((u, graph[u][v]['relation'], v))
            tmp_rule.append(graph[u][v]['relation'])
        result_paths.append(tmp)
        rules.append(tmp_rule)
    return result_paths, rules



def apply_rules(graph, rules, srouce_entities):
        """
        result:[[[(e1, r1, e2), (e2, r2, e3)], [(e1, r1, e22), (e22, r2, e3)]], ....]
        """
        results = []
        length = 0
        # pdb.set_trace()
        for entity in srouce_entities:
            # pdb.set_trace()
            for rule in rules:
                res = bfs_with_rule(graph, entity, rule)
                # pdb.set_trace()
                length += len(res)
                # if len(res) > 0:
                results.append(res) # [[('Leo Howard', 'film.performance.actor', 'm.0cs72ly'), ('m.0cs72ly', 'film.film.starring', "Aussie & Ted's Great Adventure")]]
        # pdb.set_trace()
        if length > 70:
            new_results = []
            for index in range(len(results)):
                # pdb.set_trace()
                len_i = len(results[index])
                legal_len = int(len_i/length*70)
                random.shuffle(results[index])
                path_list = []
                for index, p in enumerate(results[index]):
                    if index < legal_len:
                        path_list.append(p)
                new_results.append(path_list)
            return new_results

        return results


def trans_path(reasoning_paths, rules, entities):
    """
    reasoning_paths: [[('Italian people', 'people.ethnicity.includes_groups', 'Latin European peoples')], [('Italian people', 'people.ethnicity.includes_groups', 'Europeans')],
    rules: [['people.ethnicity.includes_groups'], ['people.ethnicity.included_in_group'], ['people.ethnicity.languages_spoken']]
    return: dict, {p1: [[e1, e21, ...], [e1, e22, ...], ...], p2:[]}
    """
    dict_of_path = defaultdict(list)
    # pdb.set_trace()
    for index, topic_entity in enumerate(entities):

        for i in range(len(rules)):
            rule_key = tuple(rules[i])
            one_rule_entities = reasoning_paths[index * len(rules) + i] # [[(e1, r1, e2), (e2, r2, e3)], [(e1, r1, e22), (e22, r2, e3)]]
            # pdb.set_trace()
            if len(one_rule_entities) == 0:
                continue
            entity_list = []
            for item in one_rule_entities: # [(e1, r1, e2), (e2, r2, e3)]
                temp_list = []
                for i in range(len(item)):
                    h, _, t = item[i]
                    temp_list.append(h)
                    temp_list.append(t)
                set_list = list(set(temp_list))
                set_list.sort(key=temp_list.index)
                # pdb.set_trace()
                if len(set_list) != len(rule_key) + 1:
                    continue
                entity_list.append(set_list)
            dict_of_path[rule_key] = entity_list
    return dict_of_path


def check_prompt_length(prompt, list_of_paths, maximun_token=4096):
    '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
    all_paths = "\n".join(list_of_paths)
    all_tokens = prompt + all_paths
    if len(list_of_paths) < maximun_token:
        return all_paths

    else:
        # Shuffle the paths
        random.shuffle(list_of_paths)
        new_list_of_paths = []
        # check the length of the prompt
        for p in list_of_paths:
            tmp_all_paths = "\n".join(new_list_of_paths + [p])
            tmp_all_tokens = prompt + tmp_all_paths
            if len(list_of_paths) > maximun_token:
                return "\n".join(new_list_of_paths)
            new_list_of_paths.append(p)


def construct_entity_prune_prompt(question, paths, neighbor_entity_list):
    # pdb.set_trace()
    relevent_information = ''
    for adjust_truple in neighbor_entity_list:
        h, r, t = adjust_truple
        # entities = h
        path = h + ' -> ' + r +  ' -> ' + t + '\n'
        relevent_information += path
    paths_str = '\n'.join(path + ',' for path in paths)
    entity = h
    # pdb.set_trace()
    # print(entity)
    prompt = score_entity_candidates_prompt.format(question, paths_str, entity, relevent_information, entity, entity)
    # pdb.set_trace()
    return prompt


def parse_score(result, candidate_entity):
    match = re.search(candidate_entity, result)
    if match:
        score_list = re.findall(r'\d+\.\d+', result)
        if len(score_list) == 0:
            score = 0.9
        else:
            score = float(score_list[0])
    else:
        score = 0.9
    return score


def entity_search_prune(graph, dict_of_paths, paths_scores, question, model):
    """
    graph: G
    dict_of_paths: {p1:[[e1, e2, ...], [e1, e22, ...]], p2:[[]], ...}
    rules: 
    """
    path_entity_score_dict = defaultdict(dict)
    # pdb.set_trace()
    a = []
    # for path, path_entity_list in dict_of_paths.items():
    for index in range(len(dict_of_paths.keys())):
        path = list(dict_of_paths.keys())[index]
        path_entity_list = dict_of_paths[path]
        # path_score = paths_scores[i]
        # pdb.set_trace()
        path_len = len(path)
        if path_len == 1:
            # pdb.set_trace()
            one_path_entity_score_dict = {}
            path_score = paths_scores[index]
            for one_path_entity_list in path_entity_list:
                one_path_entity_score_dict[tuple(one_path_entity_list)] = path_score
            path_entity_score_dict[path] = one_path_entity_score_dict

        if path_len > 1:
            one_path_entity_score_dict = {}
            for one_path_entity_list in path_entity_list:
                # pdb.set_trace()
                path_score = paths_scores[index]
                for i in range(path_len):
                    if i == 0:
                        # pdb.set_trace()
                        topic_entity = one_path_entity_list[i]
                    if i > 0 and i < path_len:
                        intermediate_entity = one_path_entity_list[i]
                        neighbor_list = graph.edges(intermediate_entity, data=True) # [(e1, e2, {relation:r1}), (e1, e3, {relation:})]
                        neighbor_entity_list = []
                        for item in neighbor_list:
                            h, t, r = item
                            r = r['relation']
                            candidate_entity = h
                            neighbor_entity_list.append([h, r, t])
                        # pdb.set_trace()
                        if len(neighbor_entity_list) > 60:
                            random.shuffle(neighbor_entity_list)
                            neighbor_entity_list = neighbor_entity_list[:60]
                        prompt = construct_entity_prune_prompt(question, path, neighbor_entity_list)
                        # pdb.set_trace()
                        result = model.generate_sentence(prompt)
                        # pdb.set_trace()
                        score = parse_score(result, candidate_entity)
                        path_score = path_score * score
                # pdb.set_trace()
                one_path_entity_score_dict[tuple(one_path_entity_list)] = path_score # one_path_entity_score_dict: {(['Angelina Jolie', 'm.09k3mfq', "Critics' Choice Movie Award for Best Actress"]):score, ...}
            # pdb.set_trace()
            
            path_entity_score_dict[path] = one_path_entity_score_dict # {('award.award_nominee.award_nominations', 'award.award_nomination.award'):{(['Angelina Jolie', 'm.09k3mfq', "Critics' Choice Movie Award for Best Actress"]):score, ...}}
    # pdb.set_trace()
    return path_entity_score_dict   



def constract_reasoning_prompt(retrieve_path_with_socres, question):
    # pdb.set_trace()
    path_info = ''
    prompt = ''
    for path, candidate_entity_with_score_dict in retrieve_path_with_socres.items():
        # pdb.set_trace()
        for entity_list, score in candidate_entity_with_score_dict.items():
            one_info = ''
            for i in range(len(path)):
                if i == len(path)-1:
                    one_info += entity_list[i] + ' -> ' + path[i] + ' -> ' + entity_list[i+1] + '\t' +'Score:' + str(round(score, 3)) + '\n'
                else:
                    one_info += entity_list[i] + ' -> ' + path[i] + ' -> '

            path_info += one_info
    # pdb.set_trace()
    prompt = reasoning_prompt.format(path_info, question)

    # pdb.set_trace()
    return prompt
