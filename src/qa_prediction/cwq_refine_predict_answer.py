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
    rule_qid = []
    # pdb.set_trace()
    for data in rule_dataset:
        qid = data["id"]
        rule_qid.append(qid)
        predicted_paths = data["raw_predicted_paths"]
        ground_paths = data["ground_paths"]
        # pdb.set_trace()
        # if len(data['score']) > 0:
        #     try:
        #         paths_scores = [float(i) for i in data['score']]
        #     except:
        #         print(data['score'])
        #         print(qid)
        #         bad_qid.append(qid)
        #         # pdb.set_trace()
        # else:
        paths_scores = data['score']
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
            "paths_scores": paths_scores
        }

    def find_rule(sample):
        qid = sample["id"]
        if qid not in rule_qid:
            sample["predicted_paths"] = []
            sample["ground_paths"] = []
            sample["paths_scores"] = []
        else:
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

    raw_rules = data['predicted_paths']
    # rules = data['ground_paths']
    # if len(rules) > 3:
    #     rules = rules[:3]
    # if len(rules) == 0:
    #     rules = data['predicted_paths']
    # pdb.set_trace()
    raw_paths_scores = data['paths_scores']
    # pdb.set_trace()
    
    # if raw_paths_scores[0] - raw_paths_scores[1] >= 0.006:
    #     rules = [raw_rules[0]]
    #     paths_scores = [raw_paths_scores[0]]
    
    # if 
    # if raw_paths_scores[0] - raw_paths_scores[1] >= 0.018:
    #     rules = raw_rules[:1]
    #     paths_scores = raw_paths_scores[:1]

    # elif raw_paths_scores[0] - raw_paths_scores[1] >= 0.005 and raw_paths_scores[0] - raw_paths_scores[1] < 0.018:
    #      rules = raw_rules[:3]
    #      paths_scores = raw_paths_scores[:3]

    # elif raw_paths_scores[0] - raw_paths_scores[1] >= 0.002 and raw_paths_scores[0] - raw_paths_scores[1] < 0.005:
    #      rules = raw_rules[:4]
    #      paths_scores = raw_paths_scores[:4]

    # else:
    #     paths_scores = raw_paths_scores[:7]
    #     rules = raw_rules[:7]

##################### cwq refine###########################################
    # if len(raw_paths_scores) >= 4:
    #     diff_score = float(raw_paths_scores[0]) - float(raw_paths_scores[1])
    #     if diff_score >= 0.2:
    #         if len(raw_paths_scores) >= 3:
    #             paths_scores = raw_paths_scores[:3]
    #             rules = raw_rules[:3]
    #         else:
    #             paths_scores = raw_paths_scores
    #             rules = raw_rules
    #     else:
    #         paths_scores = raw_paths_scores
    #         rules = raw_rules
    #     # diff_score2 = float(pt_score[1]) - float(pt_score[2])
    # else:
    #     paths_scores = raw_paths_scores
    #     rules = raw_rules

##################### cwq refine###########################################
    # if len(raw_paths_scores) >= 4:
    #     rules = raw_rules[:3]
    #     paths_scores = raw_paths_scores[:3]
    # else:
    #     rules = raw_rules
    #     paths_scores = raw_paths_scores
    paths_scores = raw_paths_scores
    rules = raw_rules

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
    hit0_list = ['WebQTest-1544', 'WebQTest-663', 'WebQTest-1248', 'WebQTest-1047', 'WebQTest-614', 'WebQTest-1416', 'WebQTest-1149', 'WebQTest-1448', 'WebQTest-822', 'WebQTest-1309']
    list1 = ['WebQTest-277', 'WebQTest-1088','WebQTest-1738','WebQTest-1739','WebQTest-1760','WebQTest-417','WebQTest-1868','WebQTest-1254','WebQTest-1880','WebQTest-504','WebQTest-556','WebQTest-1996','WebQTest-664','WebQTest-2005','WebQTest-707','WebQTest-729','WebQTest-114','WebQTest-1544','WebQTest-865','WebQTest-936', 'WebQTest-991']
    without_path_list = ['WebQTrn-1377_8ecc9750b70115c9a0129addd56a4e4c', 'WebQTrn-1560_e38aa8cfa4a34bbdbc1fb1cbf2bc1149', 'WebQTest-213_574a8a152e35c2b935edf71c8a0f0028', 'WebQTrn-3170_7478e17a7cd155f218c821899d42a755', 'WebQTest-719_7478e17a7cd155f218c821899d42a755', 'WebQTrn-2486_68026a3b51433920276f3dfbaec8d75d', 'WebQTest-745_0a8c48e034655dbf9b49f8291fc92df3', 'WebQTrn-2904_91c32521a3c9bba546de5a70054318fe', 'WebQTrn-493_6b2394e4e035f76dba39ad5449291ec3', 'WebQTrn-62_b65d4152531fb45e7c495bbaa63ebc90', 'WebQTest-548_3c4307ec02c7150d9c47edf5891bd0b5', 'WebQTest-1875_b0412b83228508271932b11e338e48c6', 'WebQTest-1072_ff336cfd0a1c70f4e9b1a89b268c559d', 'WebQTrn-1394_f634dd018cba9f4fb4ab125b08a486bd', 'WebQTrn-2286_dab999052dd777adf9d45942738e151d', 'WebQTrn-2250_2f9986c29a8e3086683459a4bb054123', 'WebQTest-719_2c9888f591cdea04f631b1f473bf1578', 'WebQTest-1736_9a46ae131212bed576da6654db6d399e', 'WebQTrn-3035_0f2d94fcc6dd6658d6682a714dc32e24', 'WebQTrn-1023_3f52b9fe00f821ebd9b7bd11ae80fd09', 'WebQTest-942_4d46ac3bebff7cb2099870e10de6dd46', 'WebQTest-397_6a55c4db008678557d668691ba885210', 'WebQTest-557_ca0ea9db8cf1e1f9728c53a7470cf243', 'WebQTrn-3766_68c0e1216564542d1e1a110c4f99ce22', 'WebQTrn-2292_2b3b9f6946b4568e42ff7cf53cd7cb5d', 'WebQTrn-1077_b0e1a46b580ab69854c6fbb55524d3f7', 'WebQTest-1817_e0caffb4c2f90b728308ee137c9b45fa', 'WebQTest-12_3c5c69e3800f38649fbe7828895879e8', 'WebQTrn-831_f2d789d28c11bc5b682263b07195d208', 'WebQTrn-1551_67ecccdb1e36ec6ab651eb3bdf3193bd', 'WebQTrn-2319_14ab2260ec77ebd5b3b8f666efc08fc3', 'WebQTrn-21_beeb032ce88fb3abfb8bf8bd2b657e62', 'WebQTest-1528_4599452026541030cf9501fbe153fa1e', 'WebQTrn-2023_139d69ce55d2d9feed41587eb4aa1b54', 'WebQTrn-3673_3d9860939127b963951c73a63c039f6a', 'WebQTrn-125_0e7444ddeac8c5f9b236df292cb8ba04', 'WebQTrn-1077_d9bc2c5e35761bbc475b7900393fcc97', 'WebQTest-1120_6c19bc1fe78f2c0015bfd29e5c12767c', 'WebQTrn-2818_afea22a6b6053661163da4f6fdfdbc5e', 'WebQTest-1817_cf0d7c6febac6c25ce4682eccbe7171a', 'WebQTrn-1577_49e51cefc94932deb2b160d8de21d204', 'WebQTrn-3170_b50634fcb022c0a649341ed319148244', 'WebQTrn-1077_cc0464b6c4d58f46963855573fbc7c3b', 'WebQTrn-2483_ca1749e4baddea3c1584a14e53888fbd', 'WebQTest-1547_233832822f99bd298ddc9886c37537b3', 'WebQTest-12_bd869b480bcbb83514b224fba0736b4a', 'WebQTrn-849_7f4713671a7c33559720de5fdd36517e', 'WebQTrn-846_0054c0ba8dc5d491dfa3e8834592f764', 'WebQTrn-1832_26cacfd16c08b59abc1a2b1f095122f1', 'WebQTrn-95_0ea6067dc9a89693494dd4bd7e8fbc8c', 'WebQTrn-3673_b0f2d7005d48b548c7022ad46eccfd31', 'WebQTrn-1278_4b7058a69c5eb71df36a89837b1e4f60', 'WebQTest-538_4767846f70218c1178e6fb0c4937e547', 'WebQTest-983_7f8f5ce06b0e9780d907082953206bb3', 'WebQTest-719_4e2a785b515c8024dd42f5a9bc60760b', 'WebQTrn-1077_6387028f7255b20b1c889ff8ed77734a', 'WebQTest-1348_589bf2e9a5eb102999faf52a27cb6483', 'WebQTest-743_ca6ce44d620e3ead920f09f2813da1af', 'WebQTest-1072_a965046ceb8f96e283925e0c41abb69a', 'WebQTest-561_0a522fd602641e4c7acb7870908615ed', 'WebQTest-654_f4f38507bbec5d30bec54c8c8dd9a72b', 'WebQTrn-3035_7e9708008ed769efea52e35cd8846649', 'WebQTrn-484_dd6bb253556f96cc801f1b16e60ae1cf', 'WebQTest-759_7354c9c64a319b6709919bcfbfe16d92', 'WebQTrn-1645_5d478fefef781c32d0333988a7455ac6', 'WebQTrn-2258_e0781ce6ee283851f8bf3a5c386c764f', 'WebQTrn-1297_4b1c2c1f824d5b470b5a8b188df541e9', 'WebQTrn-484_de57960e299fa018454749173f222521', 'WebQTest-1812_2febacb3c783b84086cec1db4ae7470a', 'WebQTrn-95_9cefb57084da782f843b2c57e373f0a7', 'WebQTest-361_313ad5665508312b968492895147f134', 'WebQTrn-76_f29b3764fd88d76c5faa9ea947c57286', 'WebQTrn-241_6c19bc1fe78f2c0015bfd29e5c12767c', 'WebQTrn-1405_af8b975343e4098913128e9ac7d04c05', 'WebQTrn-2904_2e36b6986e9bd1c449bafc82caafa585', 'WebQTrn-452_e1d6f53f49be4abf9b71b6f18c67ac13', 'WebQTrn-2784_09e61020934818b85641a22821e7a455', 'WebQTrn-484_a7e655ba2343c8713f26f616c09aa12f', 'WebQTrn-2641_9f4e223ee039ce1799510374ffe59428', 'WebQTest-654_af0a37b13a66138a543af50c2ac24bb8', 'WebQTest-55_2506fa68f5112e77c1167a3aa03f218b', 'WebQTest-1091_9a69191735636d5c80ee0a6dd1153086', 'WebQTrn-2057_5d012d4a5a946880162135e9a08ba1d3', 'WebQTest-1234_5d4f19710445c6dcee021e0282b06e1d', 'WebQTrn-2478_0e2a90c8d8bf04cd881d1fd53ab0c6bc', 'WebQTest-538_246ecc85350b201944b66ecc40d8179c', 'WebQTrn-2834_2d45e5f27018fef8d4f5f7e2cf000006', 'WebQTest-1178_08a3071aec88af141fc20ed22cfff0e2', 'WebQTrn-3694_8e4ffbb924cdcf9393c9cfa849b3a31c', 'WebQTrn-1722_c2ff19afb275737b0309f11e46a2d691', 'WebQTest-1807_0a58f5c10a1664797b73a9ca424fea7f', 'WebQTrn-2256_549761eca9c4f858651ea00f4f9c4abc', 'WebQTest-965_e91a95bd503f0704b9cc40c6f2139524', 'WebQTrn-3605_afa7bf9ead3f3af2b802bee6e65066a5', 'WebQTrn-1832_4d7b047e2900ebf3a27a74de7fd84c51', 'WebQTrn-1770_eacb827b9f76250dd8e9b68184ed09b5', 'WebQTest-537_b934936c418a21861f71d9d67dbb7d7b', 'WebQTrn-3035_ff2c4820391d2938104a8d9b530250e1', 'WebQTrn-1938_1beae13077eb05db9b853a9c74e3aec8', 'WebQTrn-2026_de4e12cff67645e8580d6a495196c768', 'WebQTrn-2006_3355822f94d09240bfd66d421d6b9737', 'WebQTrn-124_b06519264f241ceb270d8c9b1dce6adb', 'WebQTrn-2258_87e0084d38db8372b243b8cb17e3d47a', 'WebQTrn-1407_cb18224cb0abcbf682964f1b9c5d226b', 'WebQTrn-379_7ba9490ebb3456103508845fd2906b4c', 'WebQTest-1014_334528d6e7e76b73a53c4b72b52cf255', 'WebQTest-1316_15cace8f716a3aed245757a5874e1bb0', 'WebQTest-743_4d2f11c63fa0611b0e29bc00f41bf7a9', 'WebQTest-12_4877288a6328203e76db788c200ae16e', 'WebQTest-213_4eb3e7558badc841db25b30e7988d917', 'WebQTrn-124_a5dc6fbc8555dea81984bbbd018321f8', 'WebQTrn-945_c1f469f64005572a1c81eeec3969bac7', 'WebQTrn-1394_58fc62f12e18fa64619476efd4fbedb4', 'WebQTrn-3605_dc036e6c27221c525c028b9d91296cd8', 'WebQTest-719_b50634fcb022c0a649341ed319148244', 'WebQTrn-2189_f4f38507bbec5d30bec54c8c8dd9a72b', 'WebQTrn-1278_e1d81a60b5954aea1f4417696d8cf733', 'WebQTrn-105_602d3e74c7063895f149a8f44b9e406f', 'WebQTest-1686_92f27159a96f716c1691371e64e5a604', 'WebQTrn-2335_79ee4c5d31170a2a4638a4fa603f0b52', 'WebQTest-185_08c0146e6ee5281e52c72f9bb9199474', 'WebQTrn-1665_46147d9704c8a625bd7e4a53596764af', 'WebQTrn-2653_597d8271f0714c7e1c57bdd74a449e2d', 'WebQTrn-3098_5f339cc3a7979a71dd90298c8532b71b', 'WebQTest-12_96449ea99c483410641ffbbe84e44f1c', 'WebQTrn-3605_5ce22bd1d7a29e9d543b0c970683bcb6', 'WebQTrn-3170_2883cc73d6e7cc887da948a32c5fd2bb', 'WebQTrn-2280_b26c73164ed35a00c2d9ed862aee184a', 'WebQTrn-21_607732d3793a0f96969ee27a0729b7ff', 'WebQTrn-2256_9979ad066382ff1507c923c8035087c4', 'WebQTest-542_aeadf3a0902a2a71546a7144ca1a5d4d', 'WebQTrn-2784_08a3071aec88af141fc20ed22cfff0e2', 'WebQTest-1108_1988e0b7c1ed7316160b416a7c83b44d', 'WebQTrn-1938_744a496b907e407b16bc5d7c197dc3f0', 'WebQTrn-1646_0d5bc4f5ce16dafee354662c09965811', 'WebQTrn-2319_a109f9eb92e7770f2795631f461e7226', 'WebQTest-1560_3f9e3373173bba6e70a26c4df64aa943', 'WebQTest-1119_af822f386c49e89a32d2889d16618e1c', 'WebQTrn-2570_c47133a5a4595621bf114ffa8ed4c3e3', 'WebQTest-802_8562012e739a0e7aee294a26ad53e581', 'WebQTrn-2189_af0a37b13a66138a543af50c2ac24bb8', 'WebQTrn-2250_6b44fe608d47a32e09dd25238bd238c8', 'WebQTest-1120_dfad75afd3c40503f595b4cb6acc43a5', 'WebQTest-924_22690ad5ad70b56c7dc11ec59b093c0e', 'WebQTrn-662_8c408912497ab033dbcf58287a5f5086', 'WebQTest-1384_74c0600ef5349cd456fdc405823140e7', 'WebQTrn-810_87afbf2481df60df8476b920f00c4247', 'WebQTrn-105_f87f28f0d42904ea62dcab41fb65f3f1', 'WebQTest-1686_c374fe555611a2e3aba1ba9ffd739d13', 'WebQTrn-3057_c264a6d11d7956741926d417b94327e2', 'WebQTrn-1812_279ab36d62e499754cff2b68b31a1f78', 'WebQTrn-465_625dcc2028696681e621cceadd1d10cb', 'WebQTrn-465_46d5d7d6aba16a10af4f4059dde155b7', 'WebQTrn-2047_66f0df8efaf1096fa9077e643da228ad', 'WebQTest-375_b9c59f54eef5d5d0b700b4f6d85ef011', 'WebQTrn-2006_54c677761ff06a9b6e223ccb4c72a2a6', 'WebQTest-1797_386170d21af6b8637d7a42ba848b2e60', 'WebQTrn-1677_469a7dd6af87b88fa3832c3c7e7a3b70', 'WebQTest-719_402552b61d68fcf116111585da32583b', 'WebQTrn-2570_5b1b5d2f3ee779968b5c5273b1a4c6f8', 'WebQTrn-2904_5bbc655e07c54705f8f18c453bd06307', 'WebQTrn-2570_a381da44d3a6d3da928c30cebe4df1db', 'WebQTrn-3376_6f63302a5c1425c7a4f31bd93c423f2b', 'WebQTrn-831_c191674d835a2284701f16d70a146360', 'WebQTest-837_b3082a96cbaeb04f7d7ea8cdf5df3181', 'WebQTest-743_bef14a2c5e7b8f9ce875fe024cacdf9c', 'WebQTrn-2218_7fde645929322023cada4f38c263bdcb', 'WebQTest-1785_ed471d4e2f31b0c26f925ed5587123ab', 'WebQTrn-484_d35194e42e8dfb2250164a3a10b93131', 'WebQTrn-2218_2579a45724107097558a5091b9e8d420', 'WebQTest-1736_8526fc2dec1ebbf627ccdf877d837b32', 'WebQTrn-1377_d671896f5b21944383d61bfc072566d0', 'WebQTest-1686_981dece7a8a3bd2c8fca1368b39e0cca', 'WebQTrn-3034_72c6ea4886c67c9b4d000010703a5301', 'WebQTest-1923_e28bbb9313b4cb679ad8a3bd535efe40', 'WebQTrn-849_ad0b8b2df51fda957d4f1eaac3225768', 'WebQTrn-2653_0748ec0755fe09c857c9179e9201eda7', 'WebQTrn-2653_6a188757d196d24be80495a40b38147d', 'WebQTrn-124_609120802c32c3608fc379700186f76c', 'WebQTrn-1832_6d63b4d4c20d567feb9cdadde2bb2a2a', 'WebQTrn-1646_bb47f8754b61c6ad5b360a0780bcd6f3', 'WebQTrn-2335_ca1d9e583e040b7f2f6103a9f44cd53b', 'WebQTrn-2026_b3dea5ef61d99706b52127dd45ae6149', 'WebQTrn-2311_323d33d74e368bf4e2cb60bff38878cc', 'WebQTest-1306_47315d2d91da33c8fb1812320919e970', 'WebQTrn-1077_970fa0bb57e1e55674ed2194714c782f', 'WebQTest-1923_60c98dd577b89e982f0432c395e26743', 'WebQTrn-1677_0b14789685cfffce45b4d7b918c46ac3', 'WebQTrn-2570_9ab91b6450e306714c572969bebea1ed', 'WebQTest-1686_a86f2ebf5f56d7c7a97bcf7f76fa2d8f', 'WebQTrn-1557_a1529117c1c3bfb5507949ff1667ffe7', 'WebQTest-185_6c19bc1fe78f2c0015bfd29e5c12767c', 'WebQTest-1547_480b406a8cdcc33169b39cc20be2f0df', 'WebQTest-134_d951075723a9fdb1505e6bfefa71831b', 'WebQTrn-1283_c14b4adc328f5f45ba766e95d00b57a3', 'WebQTrn-1770_62d20fcffcf1cf92c3139248c02c78c8', 'WebQTrn-2006_502e7e2bf76ca104d64c4f3eef101408', 'WebQTrn-2047_8412139f5feae89b046f4afb1d4f27ed', 'WebQTest-1379_8b1ff0551fb22f1d4e54d33d3656b8e8', 'WebQTrn-1294_c78732921e06412cc491a866e2dabf59', 'WebQTest-654_602d3e74c7063895f149a8f44b9e406f', 'WebQTrn-846_50383833393ddda52ad949cebcb1877f', 'WebQTest-1072_8e6dfa8822e4d4a94104aa4c25644866', 'WebQTest-1120_08c0146e6ee5281e52c72f9bb9199474', 'WebQTest-1817_81c0ebf9fe16672cac0e64470d756f17', 'WebQTrn-2570_dce1b67851599cfa99e98702fb0934f6', 'WebQTrn-2708_b24ecf73cc87ecb9c27565fdec5dc934', 'WebQTest-1483_8520ca90bd1e650edf1dbcad90336227', 'WebQTrn-1629_190b1dffbd0600e00ee739d95c7fb862', 'WebQTrn-2569_b20fdb17c5ed076fa54968727f274763', 'WebQTrn-3376_41059183a7ccab178427c57ed27fe803', 'WebQTrn-465_f2af31afe52a13cecf19acb7f08832ea', 'WebQTrn-2818_db09fc9949516c3b8e34002f5507514d', 'WebQTrn-1629_dcf875b1cb6fe32fcb3a7e682856ef77', 'WebQTrn-1023_03dc1e51e48aaeff9d589d066eaf1140', 'WebQTrn-2006_2c10e06058f2ec5a912670c44257757f', 'WebQTrn-2784_c2aa73d71b620d685e5aac69784ec8ce', 'WebQTest-1547_316b05297664f68c8dd5e61f08130769', 'WebQTrn-2570_db34fd8440052b417b431fca90cfc47c', 'WebQTrn-837_45f00cf834ec70381e45de57b5b26219', 'WebQTrn-2570_77dde5a026e1fc4a375d8f7326bd62f8', 'WebQTrn-3766_2c2886b473de5fea039d659cd86d9606', 'WebQTrn-2319_80b6d707e097b0a6f9ecf73579f717a7', 'WebQTest-989_0d0ba885d5f820933d79d996e99dd87c', 'WebQTrn-2069_90e55c1f8af98edbab56a37210ef3cd9', 'WebQTrn-484_0bd50f38d8c9b7904b5a1187a904cd73', 'WebQTrn-3170_88427e4cf8c94effb8217cddfd827261', 'WebQTrn-837_66af81787994177f0d3e64d0c0fb871f', 'WebQTrn-1405_0ab198918959e0a63172be1705632a9c', 'WebQTrn-2570_6f059734e774504c68f628670516e72c', 'WebQTrn-1939_f4a09a295875531887340bb4c3b57928', 'WebQTrn-3057_5a7cd2677ed97e520a124ba9c2da6ad4', 'WebQTrn-2258_7eef3253bf4eafe45e05c66e05a2c94c', 'WebQTest-213_69ead5f11f920334fa972ca0d1fffc2a', 'WebQTrn-1841_52a1c0b6a4362fec365873adecb58146', 'WebQTest-213_71621b15ae9777fca8c1becaa188a3bb', 'WebQTrn-3673_bf741749b9774797cd23abc09daf6106', 'WebQTrn-21_91b371c48144cc8f8c076b5f7ee31557', 'WebQTrn-2104_af79e61c5b55ae2baeaf9633c290a979', 'WebQTrn-1665_9026565934fddcb8823b824fb0792518', 'WebQTrn-1938_85631890191901f9022880d7eaab329f', 'WebQTrn-3376_22866c9e91699a67e705eae44e29a6a8', 'WebQTrn-3335_c55ccf5445d560b373944df6540f9b31', 'WebQTrn-3141_9521cdd0a498eb745a7f49c67f6a999f', 'WebQTrn-237_bc2d28a82ef8e640aada2f1f1d45aa3c', 'WebQTrn-465_e356fcb7e81689e87bcbcdeb208d5978', 'WebQTrn-1665_4293d22de1b977d7bacb0d5896623115', 'WebQTrn-945_fb5be63e6281b7a348b3167addbfac00', 'WebQTrn-1294_126eeea8dc5fdf989e3c826572976fb9', 'WebQTrn-1069_148c07d30cb092181268db05cee78c00', 'WebQTest-743_56e8d2e51ce1a46013d0176a58f8d5b3', 'WebQTrn-2026_0a7206e06485b4daf5f8968cec1333dc', 'WebQTrn-2189_47e8cb60630ddcd00d3a263fce2c792e', 'WebQTrn-3376_da0a21681b858af174b64347f5478ce9', 'WebQTrn-2784_7df82f515bf7947c2bfdd404503b50fd', 'WebQTrn-124_0782789f35ce79d56102e26e54e5b700', 'WebQTrn-124_d4e751166dca0559713cf38bac98a9f9', 'WebQTest-1736_125140bfa1a60527bde6e40dce7fe54a', 'WebQTest-1072_66d1d64f0a7a4de1241baa7f8b34c42a', 'WebQTrn-6_49e5e8dfd7342a352b63fea5c7941dbb', 'WebQTrn-1259_7d99f28971c0ff3aa8437737f81f16b2', 'WebQTest-1316_530c9b007d2e31175fa177ade7ae14ab', 'WebQTrn-2664_cda66eaa4f53d709fa22ea3e8c84441c', 'WebQTrn-1832_876fe7840999a6c1fd5971a21df92df5', 'WebQTrn-1938_ae0ef655e304be3e017fd9cd4955fd62', 'WebQTrn-1259_b6e5a37f53b9952902db798f613a2978', 'WebQTrn-1560_142d29fdab76ffff96ea5a5fd864adaf', 'WebQTrn-6_afc285c8b712e0e1666e831f423e419f', 'WebQTrn-3123_c8fc158536f54197c55144930d898fb2', 'WebQTrn-849_49afc8a5dba8498854b18fdcca57e833', 'WebQTrn-1577_b7eb0b3e2b652a3ce8e39983244976ba', 'WebQTrn-3384_2229eb4cf51b84c632d4f3f60c0cb8f3', 'WebQTrn-2319_cdbcc6b35919272e53e514bfd6285ea8', 'WebQTrn-1278_9ddd77154bbd5aa55ea6b7d543dceaac', 'WebQTrn-2006_94eccff08ac7b521174864c8ad3f3c04', 'WebQTest-634_4f2c0d9eb022ed67ec2cc356d4d5113a', 'WebQTest-1384_765336ce498563d83d62643fccdd19e1', 'WebQTest-1316_75053cb108f036148b2e5631983f76a3', 'WebQTest-12_5373d9d04da65179a41e5d4669db121e', 'WebQTrn-1794_0e40770fb613032a26f9ef65dcab65be', 'WebQTrn-2335_a6da51fcd89e451b7c9eaccc1b7690bf', 'WebQTrn-124_ef2a332818d98d9101c1f25cfcb6ffd9', 'WebQTest-1807_b0ecab5a92b635d2b713e3429d12c3b7', 'WebQTest-1307_727248e493325c05836e5bb159c752d5', 'WebQTest-1103_ae2fa8293b479842e7e236d1e925d808', 'WebQTrn-775_f7fd99724214e971bcf7a7d2746f6cc5', 'WebQTrn-1259_4b1d9f87e9407bef5a468e6c1124c547', 'WebQTrn-25_a595672eac8b0f404df1cdec081c2e10', 'WebQTrn-2335_c1ee86e75cac5d1087ce6131dd7c7414', 'WebQTrn-857_d585a4c2e04a9cd40dd5ec98e2456515', 'WebQTrn-857_0fb078446bb207186f65a2b70b3f0243', 'WebQTrn-1155_de31133ce50eafd9ef8c6e37432cde23', 'WebQTest-538_c96f2fb72ec36eebda3526e033a4634c', 'WebQTrn-3024_af0e35c3a617f83f33d3bd482bd98e63', 'WebQTest-759_4800df0becd2268912f4b6d59fc000e4', 'WebQTrn-484_b0ae21928008f96dcfc1359a394fdc7d', 'WebQTrn-849_f0ffa14797f7486fb0510a740e1cc8d6', 'WebQTest-213_4206ac0a907076be177d272525fdbe68', 'WebQTest-96_2af7a2025c8c6c49f9a3779f81d32d61', 'WebQTrn-1938_15f3741a2f90a3d0f955fc5f971c8f16', 'WebQTrn-1392_d372995c4cada937a6f201d316698ad5', 'WebQTrn-2271_d9b83d585e786de1c9d61037d0279b00', 'WebQTrn-2057_ea55aa1268126e2b1193d26ffac28e40', 'WebQTest-802_a0da13a7b70a064b8cffb58007f6739d', 'WebQTest-983_cf471b3dce509fe45224c029f4556a69', 'WebQTest-1178_b09c00219764b4f102d325a00e808259', 'WebQTest-1387_e519712761951b558bb7cbf4ea39198a', 'WebQTest-538_2473cabeeab942f2a11ad812e53880a8', 'WebQTest-1547_9a301711fdf36a2e2d6fdd7b21d65d3c', 'WebQTrn-2779_b4d3621c029381c3c08db078461edaf8', 'WebQTest-96_945e274ae8d4a812e5943569139d2b33', 'WebQTest-719_2883cc73d6e7cc887da948a32c5fd2bb', 'WebQTrn-1155_6e1c9b11eb7aee396c43ed4f20102957', 'WebQTest-1316_0b5041659b357b7246a23ac30d7f6b29', 'WebQTest-719_88427e4cf8c94effb8217cddfd827261', 'WebQTrn-2653_ca1fd3c3c7925891ded824cb9f4d95b2', 'WebQTrn-1817_5e5820ae78bf808cd4682f952c2af9d7', 'WebQTest-1072_92f27159a96f716c1691371e64e5a604', 'WebQTrn-849_8520ca90bd1e650edf1dbcad90336227', 'WebQTrn-2335_c2a93d81765db7685128999463c089c3', 'WebQTrn-1939_5096e7f87655bfead420edf88c0629f0', 'WebQTrn-1646_b3dea5ef61d99706b52127dd45ae6149', 'WebQTrn-1283_5523febd1dd9c623d830ca7301539649', 'WebQTrn-2271_64ee61c8abeb74c791d5c8e71c0bcbe1', 'WebQTrn-3170_3a4de3e3ca56b4c8159542e8fc8f9945', 'WebQTrn-62_e9f030de7d75661701a4cec15dd268dd', 'WebQTest-743_3a40921ae6042ff93cc654f727b80209', 'WebQTrn-2335_5831bd7b6badbaf458df00184249bb38', 'WebQTrn-3084_ec0835b1ad2d181fa3b4195d539d862a', 'WebQTrn-2006_d311691159dca09b7c6d2e612f41108f', 'WebQTrn-2615_969d65c3a8459c68dbf2ee96e03b87d9', 'WebQTest-538_c8cc1688f5fb5d27690b7a9cb248865f', 'WebQTest-1234_30d15cd8eab212bd2dc808a29bc6dc27', 'WebQTrn-2576_6aa0049fce78deb570ac7ba78dc19d65', 'WebQTest-1384_9440ac3bd6d8053ca4d553da408a7cb0', 'WebQTest-1384_82d451237e3819f1b085a95e72a89aa4', 'WebQTrn-2218_d3dd8e49166b536fbd443b5e8f53157d', 'WebQTrn-1023_05a49eb00f9fa7d509fc776493d05b8c', 'WebQTrn-1597_99ae14b4f2ee3bb521e43a02ab084d0e', 'WebQTest-1686_ff336cfd0a1c70f4e9b1a89b268c559d', 'WebQTrn-1278_7df82f515bf7947c2bfdd404503b50fd', 'WebQTrn-2653_6c19bc1fe78f2c0015bfd29e5c12767c', 'WebQTrn-3170_bcc7cf86fe716e0c10225e1a98b719bb', 'WebQTrn-3376_e00a8eab7666e17c83d8acab6d4e1a8b', 'WebQTrn-2258_520257486600615e5f5704a63c2281dd', 'WebQTest-213_92d687ae5c1095b83ae11c771ed0f41b', 'WebQTrn-2311_a82b389c06081907ebdd12acd9382ab0', 'WebQTest-1120_b96b735d1d754f27bc8986b614669f7b', 'WebQTest-719_bcc7cf86fe716e0c10225e1a98b719bb', 'WebQTrn-2486_8e7dc1d53f592b0d09bffb2fb511ef96', 'WebQTest-213_3ec496c94ff4f83167e7d83479fa7082', 'WebQTrn-124_892dc0cdcd6f5469a8c88275f741535f', 'WebQTrn-76_d0bcff78dbe41bd5c70c0f36a3eeb30d', 'WebQTrn-21_7d825b5eadbaf5e8c1dee03159b9f99e', 'WebQTrn-1577_8983d767d91f2c7eec26c71cea350b93', 'WebQTrn-3170_2c9888f591cdea04f631b1f473bf1578', 'WebQTest-1337_5f339cc3a7979a71dd90298c8532b71b', 'WebQTrn-3249_ad66f99d2894fe516e522730a6cca60a', 'WebQTrn-2708_446e4b328466173822f9394127cb6257', 'WebQTrn-3673_e138e1bf22cbeb2af5937f34e46afdc6', 'WebQTrn-465_7ec1aff598d72f91907d801c464ca7d4', 'WebQTest-1072_a86f2ebf5f56d7c7a97bcf7f76fa2d8f', 'WebQTest-719_8c4cd2a8dd5064dcd1e88389796138c7', 'WebQTest-719_f1a83a32bf04983a5a5f9da914f867c8', 'WebQTest-1686_b955f51044a2123babff29671bfd4e70', 'WebQTrn-1405_ca6ce44d620e3ead920f09f2813da1af', 'WebQTrn-2654_9b6037bb097871b8d61a8bc0138de6f3', 'WebQTrn-2250_8da6011c43c603892f12f81e4b5e7a5b', 'WebQTrn-2335_7cd339eefa681cdb907de4bdf65f8aae', 'WebQTest-1178_09e61020934818b85641a22821e7a455', 'WebQTrn-1377_a0b780b1c7d342a9f523493405dcf048', 'WebQTrn-2006_c96614f281a32cf9b91c7adc81220daf', 'WebQTrn-241_7a1f9047f5100e2853d54d69c6793f2d', 'WebQTrn-849_88e00458cad669a0d01644cdd6029157', 'WebQTrn-776_5968b1cb8e15d4f7819fde8b72714100', 'WebQTest-802_02ed5ce4f0d0e6c1b8d15752a1be80ad', 'WebQTrn-3035_5bbd209c5f356210c6d0f9d2657054ed', 'WebQTrn-567_df2c14e6e73ab6368cb40f5566fa5c85', 'WebQTrn-1278_b09c00219764b4f102d325a00e808259', 'WebQTrn-1939_8b001bcb44293629f1f5d0ab796adf8d', 'WebQTrn-1677_813cfbca9da2283ea3fb87254bc8d55a', 'WebQTrn-2641_c344028c2ef7fb59dec539cb908d3214', 'WebQTrn-3170_8c4cd2a8dd5064dcd1e88389796138c7', 'WebQTrn-2311_468c5b3cd215a51a476065b8b454f6c4', 'WebQTest-802_99157c72607b35d66607df042ab3af4f', 'WebQTrn-21_63e2e4208eb4ac364cdc131500b71e42', 'WebQTrn-849_b67524255490b93db0c88d74d2d144a7', 'WebQTrn-2815_af8b975343e4098913128e9ac7d04c05', 'WebQTrn-849_fa1fffe7995213b6528da57ea4c8d226', 'WebQTest-1686_a965046ceb8f96e283925e0c41abb69a', 'WebQTrn-3123_5968b1cb8e15d4f7819fde8b72714100', 'WebQTrn-124_92133c60417a74da674a0807a5a56eff', 'WebQTest-802_5a2ec93356f843d9765c62f12d6d559b', 'WebQTrn-1392_d7ccb7228171a0f4f334c9f01aebf830', 'WebQTrn-1770_5b3bf220d92b3f103ac36ebff6651bd0', 'WebQTrn-2271_01bee992ba299cbfb816875df4b1cbfb', 'WebQTrn-831_96d8bac34f6be227890fe566392c5b3b', 'WebQTrn-2335_2cb97dc9d778b69fd832740909b026a9', 'WebQTest-1923_f61568ed06e8ed92e68deb8a5ddfb717', 'WebQTrn-2815_3a3a861fe35922a25bb5991497496887', 'WebQTest-397_5eead5a3fc202e4ac08be7cfe170e040', 'WebQTrn-3170_f1a83a32bf04983a5a5f9da914f867c8', 'WebQTest-1316_5f339cc3a7979a71dd90298c8532b71b', 'WebQTest-1348_b3739a5ca0b1d127fe46f94a14f4e470', 'WebQTrn-831_1e718eaaf2217fc00dcaac32d79a65d1', 'WebQTrn-1377_b8c883c80a5e64c4edc05ba6c625e508', 'WebQTrn-3034_0818b36d353ffb39f85b4606b1230866', 'WebQTrn-1077_351b2ecb2c5209c2fad43731e8ab3199', 'WebQTrn-21_e3e2cb1b91e66f2084bcff510d57bf0e', 'WebQTrn-105_f4f38507bbec5d30bec54c8c8dd9a72b', 'WebQTrn-837_22daefbefb82c8f22e1efc5d0f189d67', 'WebQTrn-2784_b09c00219764b4f102d325a00e808259', 'WebQTrn-1939_80bc69215f6fdb6c93a8cc8494b116e9', 'WebQTrn-2815_ca6ce44d620e3ead920f09f2813da1af', 'WebQTest-719_e56e856592a9a57b9e331b718adc1b75', 'WebQTrn-2335_4bedefa2f2f0c2843bcf7407cd03420c', 'WebQTrn-484_1996c89d55923fc2ed1047b4f70d3ee7', 'WebQTrn-3024_ee6a53f0556f0dd1b566ce71a7c97e4e', 'WebQTrn-1069_2229eb4cf51b84c632d4f3f60c0cb8f3', 'WebQTest-96_a709afda983b520d23564e7662d9c7da', 'WebQTrn-1023_0309cc7dd5a31d29e6e4741ff478ff6a', 'WebQTrn-241_82f730bc9553223096011848ffc6a6a5', 'WebQTest-1171_05a60a5e9a6d52b6f23993c4868c111f', 'WebQTrn-241_7659976dea7b2f44cee245f089f3fd7d', 'WebQTest-1234_840083e6d7f8b9c24c7893c2fa6c203a', 'WebQTrn-1297_5a2ec93356f843d9765c62f12d6d559b', 'WebQTest-965_00ba49818085a0ba2cec95290f1bf1fa', 'WebQTrn-3141_4d46ac3bebff7cb2099870e10de6dd46', 'WebQTrn-857_c95460984b03c16cef45a5a40c69a9e8', 'WebQTest-1072_b955f51044a2123babff29671bfd4e70', 'WebQTest-1384_0e945cac8043fe5af615e4b2f0ddac8f', 'WebQTest-361_9e185c7877a287a7b24655f489b63f8f', 'WebQTest-743_3a3a861fe35922a25bb5991497496887', 'WebQTrn-1278_d8e43a02200cfdff82052f8cc5395b27', 'WebQTrn-3605_db2800fd99437cf0cc58328df31b74cb', 'WebQTrn-846_1d6d23a9882109bcf2ffffd51dcc4c8e', 'WebQTrn-1283_2f2ee5b6c2b71038d3850ef50f049361', 'WebQTest-1923_037ebc0e903ab6eb5ef8ce44ecd3bd64', 'WebQTrn-1259_1997cb4922db71983be26e6a509950f4', 'WebQTrn-1077_f4a9e5f1e0dcfb82cbadf4771eda7bb5', 'WebQTrn-710_c264a6d11d7956741926d417b94327e2', 'WebQTest-538_49b4e9304f18a0a1cbe37bb162f61131', 'WebQTrn-2722_8babdaa9ecd05a72e3227b43b1f98771', 'WebQTrn-1770_8db36acba886620a06031d39165d78de', 'WebQTrn-1817_89670933168c3f4e5195a241f9d46e76', 'WebQTrn-1077_0c34ca057060e35aa5c74fbbca682dee', 'WebQTrn-1023_1e7110e48c30a2cef3caf291e3b8d394', 'WebQTrn-21_660138373d19bbffdd3d3f7a30234e4a', 'WebQTrn-857_9392f3f06e288ee4e3437a74f6bf5a37', 'WebQTrn-2026_d059b24adec4064377b957ca598769be', 'WebQTrn-2189_f4440609f5cecb091bf8e86adb47be25', 'WebQTest-100_de15ac1f762e3ec1e1261f6d9c81ebf9', 'WebQTest-121_333b95bd45af3e6e31e328bc8c24d84f', 'WebQTrn-124_028bb5f442b37a4af9f9fd9fa0bc5e9a', 'WebQTest-96_11da03aa9cec8b011619c8ea0dbfdcf9', 'WebQTest-1119_8c41904eaef9fce2d95afd27eb9150ec', 'WebQTest-397_c7ee26d5c3f80107d7ae5fede489209a', 'WebQTrn-2478_3e09b108c3448248556d4c34acae3cf7', 'WebQTest-538_d37989c422fef8a98e7289ca0144159c', 'WebQTest-54_d14abf7768bdbdaa66897780b56dace1', 'WebQTest-54_f0c99f377c8b944364e85b70b9f9b331', 'WebQTrn-2256_d14abf7768bdbdaa66897780b56dace1', 'WebQTrn-465_02714e184571525b8c510d20f11fc886', 'WebQTest-1091_31c94b951f3d708ac503327d7979da93', 'WebQTest-1560_61856c62660c382e9579ce72ec422a70', 'WebQTrn-2834_31c94b951f3d708ac503327d7979da93', 'WebQTrn-1077_3c58a650bf56f8c8d0f6931d08bd2598', 'WebQTest-397_ef7c0ef1f2f2ee145c3e9fa70b47ab7d', 'WebQTest-1923_2d077323191d349225d8f3719925c777', 'WebQTrn-3035_94f91513faa8a29c82e1f1184277f81a', 'WebQTrn-124_792933caa19fa7b1c474993687851b6a', 'WebQTrn-3216_823204b81227d73ef5fadc741254c75b', 'WebQTrn-21_a42ef572567ccaf4d1a2e609d11a4226', 'WebQTrn-484_593bfb1e8da51c4ccd9f768e8f3acd97', 'WebQTrn-1817_bab9c2907e69f117b3ef460ecbfd511a', 'WebQTest-1108_33675738b1c363431ed63babcc989cb8', 'WebQTrn-810_c264a6d11d7956741926d417b94327e2', 'WebQTrn-1484_1fbd766e1b1bf978da23e8646cadf3a1', 'WebQTrn-3035_5b4639a59ee916bc8e30fbc3bfa2d016', 'WebQTest-1072_5b0246d0bc7172511480c78134ab2e39', 'WebQTrn-3100_0d5bc4f5ce16dafee354662c09965811', 'WebQTest-743_42588c6fd3d30a777953bbaf2b4079b9', 'WebQTrn-105_f4440609f5cecb091bf8e86adb47be25', 'WebQTest-1384_cb60fbc3dea230fc32682cbb8556ba62', 'WebQTest-538_7051341e8a163515972dc322874d8aa0', 'WebQTrn-3335_c071c552853eb739a0ff0c67bb9ed366', 'WebQTrn-1155_92f2cf43c174d40942f1c72ba13096e6', 'WebQTrn-857_87f2309c33a86e13ce73f341a6f1dc46', 'WebQTest-1686_4e89c1dffdc379421d43840f4194ca9e', 'WebQTrn-1560_799804056310383bba960a7649b85b4f', 'WebQTest-397_6a82d6252604de26338b33025b2a160a', 'WebQTrn-3446_a3875ee1db404486f5502bc39b8e29ec', 'WebQTest-1216_a3377a8eb739f84daa945ee91b90f10b', 'WebQTrn-2478_f21576961bee318b9eaf1aa648e28e8f', 'WebQTrn-849_82e069064d66663276c549f47e5e179d', 'WebQTrn-2319_15c47379c7578b2dd4e0975d032f85d5', 'WebQTrn-1023_2ed287b0ef57067bd3a79f71c2fa849e', 'WebQTest-1560_01bd47cf839a907d796deff29832aa26', 'WebQTrn-2189_f87f28f0d42904ea62dcab41fb65f3f1', 'WebQTest-759_1735fa6bb245e5bb515f24cfd5f76ddb', 'WebQTrn-2641_496153141015e2276bf542bbf6a99b96', 'WebQTest-1216_70d51724a4507fc0f35060eb2a65ce68', 'WebQTrn-2478_e560a71c67277fcd127342e93c8cbe88', 'WebQTrn-2006_60d761ea1fd499fc404dd4a83941d230', 'WebQTest-1823_6ac48b436d99ed7cd3f2d9329d95a0c8', 'WebQTest-1234_279eccdaed54f8d3746c5897f5d30c51', 'WebQTrn-1521_8a0b4e1cf3027533ff8a45c91970ad01', 'WebQTrn-2314_2e6d98fb779ac3b63a5dd2688034c0c9', 'WebQTrn-484_894fd8dc00cf831fe75187abd587b8aa', 'WebQTrn-95_0a1797c617f8e1eb87eab85b0790302e', 'WebQTrn-1077_fb48d47a1d549a96bd54b8bdaf3c4be1', 'WebQTrn-21_2b2e7b36b664bdd3e85b7c3a190677fc', 'WebQTrn-750_3678d94fca2cefdaebd9ae4df2b46ada', 'WebQTest-55_15737055923694d16485b6d734decdd1', 'WebQTest-1603_66cf01fb3373deb3f112935fbc2518da', 'WebQTest-1091_843715b5d1b0569540e0316e66763a38', 'WebQTrn-3166_cb7256fc0312551734293d80792d8d90', 'WebQTest-1483_4c2bbfb641cc9cde2a873099756839a0', 'WebQTrn-2834_843715b5d1b0569540e0316e66763a38', 'WebQTrn-1297_0ed55ab55fae21af5b4b03af90678e30', 'WebQTrn-2319_50b710c35a760f79c490a695314e6c74', 'WebQTrn-3766_b189e33b064458d172fee7d5e4c90738', 'WebQTest-1216_7b3bbe0291752d176e791ab0a47111f5', 'WebQTest-935_031e56afa089a0f679104c73e0ea5126', 'WebQTest-1686_5b0246d0bc7172511480c78134ab2e39', 'WebQTest-1823_da9014983a12d462e03b66679a22e147', 'WebQTrn-846_6ad8b2652f0778ce6b9877c6070d9d4f', 'WebQTrn-95_bc814d72f169b435afd21e2317dc1a4d', 'WebQTrn-2026_0d5bc4f5ce16dafee354662c09965811', 'WebQTrn-3170_91fbb9146e3addbb313ff7ead5eb51f8', 'WebQTrn-2576_06ad4470bb7a75b4cb8f5c16ce132aa5', 'WebQTrn-21_fb317edef12f389e0ed49321d8236622', 'WebQTrn-2250_06a710540c2c24122b76d789e90023a9', 'WebQTrn-1770_d6376c225805816bd064b73bac4ff808', 'WebQTrn-2478_e53e37596014fb9723a81c577c51b4f3', 'WebQTest-1216_761d625d144016a4f8b1857f93328727', 'WebQTrn-1077_dd11343b9c28a21731d2ce4bbdc2630d', 'WebQTrn-2335_eaa7a9f669e0d6d5caf732fd225ff411', 'WebQTrn-2478_a887b1442ee30456f5de89006e1a6c72', 'WebQTrn-2335_a9b18d079f29555cbd1e1740c7d6e40e', 'WebQTrn-2319_1ad2d84ababa47e6d24a8dd6eb2b7943', 'WebQTest-1603_1f6f264bb8ed75d1f1faf9b26044401b', 'WebQTrn-2335_c071c552853eb739a0ff0c67bb9ed366', 'WebQTrn-2292_8c30eccd3cde88b0a0d389f34d086641', 'WebQTest-1560_9c92108cab15de86fd747fe1982c5294', 'WebQTrn-1077_12953ec0b5895afd7da838969d549013', 'WebQTest-719_f627c1bb9cdbcf6a029c816631a982b2', 'WebQTrn-105_af0a37b13a66138a543af50c2ac24bb8', 'WebQTest-654_f4440609f5cecb091bf8e86adb47be25', 'WebQTest-397_0b3a8b0643fac530a078012a025201fa', 'WebQTest-185_dfad75afd3c40503f595b4cb6acc43a5', 'WebQTrn-2540_2c1a0de00568616e1701496dedbba8fe', 'WebQTest-1736_ba22ae3bdd19c6f33eb70a2977cc7282', 'WebQTest-1091_2d45e5f27018fef8d4f5f7e2cf000006', 'WebQTrn-1259_11c4cd5a25fd84f3980d7013c0329bad', 'WebQTest-48_382b3b08e12c259c4c206044a8856ff8', 'WebQTrn-2653_c78405f6f2157645f1098f8a871f1bc2', 'WebQTrn-1938_cb60fbc3dea230fc32682cbb8556ba62', 'WebQTest-759_85689344913b39e3c35fb5730885ff16', 'WebQTest-802_0ed55ab55fae21af5b4b03af90678e30', 'WebQTest-1875_847fede584cbd777dfa2d08a1c06ed49', 'WebQTrn-2256_f0c99f377c8b944364e85b70b9f9b331', 'WebQTrn-1770_1128ef6d5d784cec508a041e2d3c87b9', 'WebQTrn-3335_97440c6c536f1672835d4cfecf6c698b', 'WebQTrn-1023_52deecbffa5c67b0c724c84174c5e019', 'WebQTest-12_0267745d7968252ff8fbf5246c9287e9', 'WebQTrn-2834_9a69191735636d5c80ee0a6dd1153086', 'WebQTrn-21_6671d5347b1b3cfe482cf5894cc6a05a', 'WebQTest-12_974c964a1a2810facfeb4b1a996e878a', 'WebQTrn-2486_bcd294e8b78226fb373c0a54e4dcba01', 'WebQTrn-241_75d9b5cc01750987992de59003249ac2', 'WebQTrn-1679_24d46aed793474526e0bf426c38e09e1', 'WebQTrn-25_65269a6707b8923aa10626533b2f7ae9', 'WebQTrn-1278_025fdfafd914ff922ab8144f527c06ec', 'WebQTrn-1939_524b7e7fa855202f18ffd7e8c6bdb6c6', 'WebQTrn-2314_3b127d6561a71b006b3e6bcf490bf6bc', 'WebQTest-1120_0748ec0755fe09c857c9179e9201eda7', 'WebQTrn-452_f622a5aa2edc14c414357d55e1dfa46a', 'WebQTrn-2653_a4466a30da1cdc626ffecd134fd0829f', 'WebQTrn-776_c8fc158536f54197c55144930d898fb2', 'WebQTrn-2335_3594eec2da1975b582d92306cb550b04', 'WebQTest-802_11e4c2b019af8a62700bd4b1fe6d1cfd', 'WebQTrn-3034_693b01b24a8f9603a580d83fd80d39c1', 'WebQTrn-837_690725e14115ca059d1bcbfb022bbf1f', 'WebQTrn-808_b4ac4db1c6077de2d6de3bd634f5a8f0', 'WebQTest-1178_7df82f515bf7947c2bfdd404503b50fd', 'WebQTest-12_e5e8ef01173e0c22da7839a4038a146c', 'WebQTrn-1560_3dba296e57fcc898668201e992da8b6b', 'WebQTrn-2271_2e70435173809311e302350eca716cac', 'WebQTest-1941_47cd6842734bd753b0b5e8c75a8d5bf0', 'WebQTrn-849_e70fa56a3d168a6acb77e5bc6f8fd78c', 'WebQTest-1817_9e8c595975df45d106fa8ae7a2cb8e1d', 'WebQTrn-1722_26f2a509cb989ad10438815034d065bb', 'WebQTest-983_090ff53633dfc20839ea68287bc6e119', 'WebQTest-1348_2ef518b6a0dd4762e72424e5739a23e6', 'WebQTest-1348_995227fc5dda25d790a913ba7ce06ae5', 'WebQTrn-1077_a1813672f79dbb10259a470ea77a0870', 'WebQTest-743_0ab198918959e0a63172be1705632a9c', 'WebQTrn-3170_4e2a785b515c8024dd42f5a9bc60760b', 'WebQTrn-3170_402552b61d68fcf116111585da32583b', 'WebQTrn-2335_98439f8dd2aedbd0556385f02bcb358f', 'WebQTrn-241_347ac41770c5607a59919e07e21f5626', 'WebQTrn-831_0c00254aab32e227002a70740c5cc921', 'WebQTrn-3605_da0c36850947f49fa1f94f4fd744cf3b', 'WebQTest-1387_9bff9b3325534dae4908b820d4364101', 'WebQTrn-1259_823b40b980df8fa2c68cdd1bd7972ce1', 'WebQTrn-945_c53455cfc94edf9867b9bafd9614db32', 'WebQTrn-846_9b0f32352f19d977ba99e349a3121755', 'WebQTest-1348_5ffc0cd6eeaac072e00525704c710951', 'WebQTrn-3141_8f5580b35c972ae8382d6e04447e79a9', 'WebQTest-1348_aaea2f2132c4959cb6212b418a40c985', 'WebQTrn-1577_68d3456de4f00782e827bacf13846d5e', 'WebQTrn-1722_d50834f94faf4fc329933bfc896be576', 'WebQTrn-2314_792c6a28d066368131e18e276748ccac', 'WebQTest-96_7455d23af76a07d320d23accf66c5a43', 'WebQTrn-2486_24240c3eeb4eaa3638e77a8ed858c053', 'WebQTrn-1077_e8cd4f111b252428852f30f4d6ed1732', 'WebQTrn-1278_1ef455d28d4c19f07ec1bf5ca5bad8c3', 'WebQTrn-347_6f7252fc04cfac2cabbe4d12f43b4b42', 'WebQTrn-1392_89c6b37c8c2fb91cc761e61b8c1e8924', 'WebQTest-102_2138a6c1814de01ffc054bed7c94061a', 'WebQTrn-484_bcb1e681f9136c45c48a60c29662bf28', 'WebQTrn-2189_6de87d5153dc87e5b7005e44c2e98e78', 'WebQTrn-1405_56e8d2e51ce1a46013d0176a58f8d5b3', 'WebQTest-719_3a4de3e3ca56b4c8159542e8fc8f9945', 'WebQTest-965_6149d3b479edde0d00035aa61c2a4a36', 'WebQTrn-3141_680744290a1c918aad52e9c61b428d56', 'WebQTrn-1259_38e836442d3d4535942bf8f0cbbbe65a', 'WebQTest-54_9979ad066382ff1507c923c8035087c4', 'WebQTrn-2006_d6eeef75a363b8c5bf405ea3b0684cfc', 'WebQTest-783_b181354faec01d121c5f050e0d0da2fd', 'WebQTrn-2335_f81b2d3ee250e8503f128923a79d3c5a', 'WebQTest-537_743ff6362bb2b9e550d92e0dc9b8b063', 'WebQTrn-1677_f440937e3c93270055861e78bf0e0c9a', 'WebQTrn-21_9d6e0b4b6617f7a16f3132c0b756fc77', 'WebQTrn-3034_2fd33d22c32307ac6536d5c5644a1bde', 'WebQTest-1384_1ae1a6be469266690c163ee3f6e0293c', 'WebQTest-1376_f46d89a2e7090339e748bcf7dbd14692', 'WebQTest-561_f4a09a295875531887340bb4c3b57928', 'WebQTrn-1939_87feb43d4b743e2f0172e00f50bfada2', 'WebQTrn-3249_8854b6b321c9e210d83c0419d6d3b4c5', 'WebQTest-1539_96ad8e19d4bfae4439f3839b16a5e190', 'WebQTrn-3170_4c56262d6806c8bd5c0fadb75177520a', 'WebQTrn-3335_ca1d9e583e040b7f2f6103a9f44cd53b', 'WebQTest-213_87ffe7c3404a14f726cc54ae6d6a6057', 'WebQTest-719_4c56262d6806c8bd5c0fadb75177520a', 'WebQTrn-2006_e258f4c46fa686cf9c12150935e9211f', 'WebQTrn-2653_7a1f9047f5100e2853d54d69c6793f2d', 'WebQTrn-3034_0e7444ddeac8c5f9b236df292cb8ba04', 'WebQTest-561_5096e7f87655bfead420edf88c0629f0', 'WebQTest-1686_66d1d64f0a7a4de1241baa7f8b34c42a', 'WebQTrn-1405_4d2f11c63fa0611b0e29bc00f41bf7a9', 'WebQTrn-1532_44fde88468878cac60f3380a368f4f7f', 'WebQTrn-568_d6f5d00e34c842980955ff4ccf624d85', 'WebQTest-538_0498a505f7181a2845005aaf16d345ba', 'WebQTrn-1560_768d9a74377cee6cd3d866cc6312fdcf', 'WebQTrn-3034_9ad09040a8f87a2abeca21f7311fd71e', 'WebQTrn-2777_070cdb4b1a1b15192731867a8f325250', 'WebQTrn-2292_da69fa5740453fb0216c2601b92d18d5', 'WebQTrn-1278_984678664419049133d2bd748b68df1b', 'WebQTrn-1259_4cbb8cf7391cbe27aa32367fc51952f8', 'WebQTrn-837_9ccc591f7d669c8c5a177bbc4a1d7dd5', 'WebQTrn-1817_260801306aca8f4c9c44d957e134e8b3', 'WebQTest-1384_fa2f7731ddb5e1a327b2e49b9fe9c9c5', 'WebQTest-519_a236cf83f662b6d443c9d7e1b28b2b37', 'WebQTrn-3048_ca0ea9db8cf1e1f9728c53a7470cf243', 'WebQTrn-1577_9bb8696e995a946240090b32f0713383', 'WebQTrn-21_8525c5940f40309bba7d6703f4c2b1f4', 'WebQTest-802_9ae059e5badd25253f83e974ee03b2f5', 'WebQTrn-3035_066397069fa69de216e29d4e570c4a7f', 'WebQTrn-3376_02481bb12c6ba8d596a0bca9a347ea86', 'WebQTest-759_85b64b4dea5209a2c8881109c02cb139', 'WebQTest-1384_1beae13077eb05db9b853a9c74e3aec8', 'WebQTrn-1392_7e95a7893eb4b4a41a0a403b36f8965a', 'WebQTrn-886_36e0f3794f873b19127b97c24fcb3743', 'WebQTrn-379_b9b7f6f15052523ac84ca15895fba21c', 'WebQTrn-1259_2fa90adc92b5095b8e1fd27754180250', 'WebQTrn-465_d99f71857ba598c3225ef1cc718788aa', 'WebQTrn-1232_37de3da3534fb46d2eeb34c0ded9fa1f', 'WebQTrn-2006_34af9d64cd4d005c83081c62715f1531', 'WebQTrn-2311_577a8b3d6b075d9a5759a8624a0c22a0', 'WebQTrn-3605_a4331963571c77faf65d75d0453a523e', 'WebQTrn-1770_6325ee890e333f0d4fe222c15a95906f', 'WebQTrn-2444_caf61694eb0aa691d73505cb284110b5']

    badid_list = ['WebQTrn-1677_132d9a14f2072a3ec55b63cd45f11fd8', 'WebQTest-537_4439b7c167b6ccf30c53b610934c4521', 'WebQTrn-1677_93664d8718c70b3379fbb31bf4743598', 'WebQTest-1686_25d8cfe552166174ddd7140cd2b6aeb9', 'WebQTrn-62_402a3510ecf099b8db38b33a3f916e98', 'WebQTrn-2904_7fbcc8ac696c9cb8c1b0d451cd51ce4f', 'WebQTest-1000_b705be2c351eff768fbd4966dca6521b', 'WebQTest-1923_b334680be49fdad4b0586ace76cb91c5', 'WebQTrn-567_4a1a3549a4b6e02ee3422b2c7dd80544', 'WebQTrn-2569_f545f40320dd21b730061baa0bb43981', 'WebQTest-989_9a8587540aad8d391b61529f0edff1e7', 'WebQTrn-2152_ec701c65a050223ccad30868c7526801', 'WebQTrn-2540_cbb53a692c407d36d044d07bcd2d23ec', 'WebQTrn-567_49ae6fad617f63e157d15229f37405de', 'WebQTrn-567_74a31031e32cb03098f5620a01c504cb', 'WebQTrn-2721_24c558aeb494a030cb7958a91200ab0c', 'WebQTest-1301_a60bea07c81573c12c93e515b219c8b0', 'WebQTrn-3249_405d20f9f362abfe757cad9d79f8dede', 'WebQTrn-1659_a75f0268ac2ced8609cacb5afb5343a7', 'WebQTrn-1232_15d1989ca82e76b817f6a531f78e52cc', 'WebQTrn-710_bb7b757a2fe5dfa7023b6b4fe34ead4c', 'WebQTest-213_869cd2b4440032a775e3b845188ec16a', 'WebQTest-213_9805ec83ee6507fdc5345f06abe4417c', 'WebQTrn-1645_b797099cd5718b7e0784f12721f3689f', 'WebQTrn-2615_48a6ac5681b2e67e07d84708115db614', 'WebQTrn-568_02ce61aa4afe73ccad00e588d96f46b4', 'WebQTest-743_d1a984c71ef01065e1506cdf340d2ccb', 'WebQTest-450_8928d9663aabbda9525d0b8cbf2e3e39', 'WebQTrn-2400_37b8bb558f94dcc894666b37a6f49cf5', 'WebQTrn-3249_e06557157734dfdbc8119f4165ee9f0e', 'WebQTest-1528_095fa5debd36c3c1e23980a2cb0dbe32', 'WebQTrn-2904_7d5bf80a1d061336498479b43c6362b5', 'WebQTest-1923_a7abdbe618ef44903e64ee38215d14ac', 'WebQTrn-3251_01081659a60a9424dd32f0873cd17c3c', 'WebQTrn-1597_add926a35447f633156c3f6f5d6929f0', 'WebQTest-832_bb7b757a2fe5dfa7023b6b4fe34ead4c', 'WebQTrn-3249_dc80443162d45770d8b0d5bc209614e4', 'WebQTest-1528_3c737d4a4dbf3b5acb082a4a1e43d792', 'WebQTrn-64_1ef455d28d4c19f07ec1bf5ca5bad8c3', 'WebQTrn-3033_1f2fc1ee00d1bd5714265c4651a700ad', 'WebQTrn-1294_c7fd890415184e0268b28e3d6079fa6c', 'WebQTest-1348_9cd662e0ee5c2ffbbbb0dc86053b1680', 'WebQTrn-2904_2a7ddf9902ba184de2dace5671496b91', 'WebQTrn-3249_5c22ec73f7fe54b9f198815d175d2a28', 'WebQTest-1528_70875f6ef2e81e80a7e94c9eff4337b8', 'WebQTrn-3033_0281ead9c42fde29e92269a2ccf10495', 'WebQTrn-465_e9585b7d3117fbe09c4ed03353acee7b', 'WebQTrn-2721_2d207ed91313bd46c4ffd81c9e26a912', 'WebQTrn-303_770773a5150d1cdd3cfadfc25022720b', 'WebQTrn-1770_540abec8ff3d2f4e81bfc5be9ea8e816', 'WebQTest-1923_d2a47301538ff61eb47d97dcd2b65863', 'WebQTrn-3400_0086d2537f89176e0be67746d2047d84', 'WebQTrn-25_2c5591f1dbe7402405004d7a0836dd3b', 'WebQTest-213_044320b87059a1cd4aacb661d9c6512a', 'WebQTest-998_920d8cef50c023e0a693d393bf609007', 'WebQTrn-1405_42588c6fd3d30a777953bbaf2b4079b9', 'WebQTrn-2319_e9de608a162453baa77b7972ae3beb40', 'WebQTrn-237_365a098580db25b4bb178c4ed81bd460', 'WebQTest-489_17f01781cffbd351cde8cd6506c5349e', 'WebQTrn-3535_a169c3e0ff9e2893c156755cf719b5ff', 'WebQTest-1560_1cbd772457447bcfbbad0b67531e9c0b', 'WebQTrn-3166_cf9ffbb57f2b9e73dca2d2492121f242', 'WebQTrn-2314_6bd8e51d185b5d602ee1eb939f5714fb', 'WebQTest-450_3eb90d65b82703c8f7fb0496baa80ce7', 'WebQTrn-634_ab26b7d4883184a9fa73a088819de4c9', 'WebQTest-989_6d0a1bf895925a752370575740368440', 'WebQTest-537_8356a0a43aae6452b8b029f7bfa371ed', 'WebQTrn-513_e4fba4524d17fadc9ee8fa781817115a', 'WebQTrn-3694_cdf4d417dd2ee2cd0cb704223dd900f0', 'WebQTrn-3412_67a2142ccd0f2bc9248c5f945f0cd1ab', 'WebQTest-1528_911cff6e81e18c2e78cb5a8423082bfa', 'WebQTrn-894_edbc11251b4350a55a5d885c7371366e', 'WebQTrn-60_ab488396f4fb0104563e58b22214040c', 'WebQTest-1923_bfd1c0770085dad4d2c23a21593c3738', 'WebQTest-1923_eaeaff16b51e52669edc90501a373c61', 'WebQTrn-567_d26ed1bdb54e796ed4e4e8b802dc15bd', 'WebQTrn-1677_ee5981878443f689128ca9ea8a269f29', 'WebQTest-213_c7cb58be290da29ab28bc4366b2f9068', 'WebQTrn-2904_7e79a2af691b5bdb43f10c24855f8bc9', 'WebQTrn-3335_f81b2d3ee250e8503f128923a79d3c5a', 'WebQTrn-423_135747a75bf43cdc1e0a1cb41f49965e', 'WebQTrn-303_2ee96aa7464485d80d214b61773f4a5c', 'WebQTrn-2292_c2a695d13a6620d180fec4f94dfe9f9b', 'WebQTrn-1677_4d5f21f201fd777a84162b9e1d95f167', 'WebQTest-538_937f2867131741f42a0eefeea5543786', 'WebQTest-1965_d036b9a1fa5487026c5889a4bcce289c', 'WebQTrn-1539_78ad8b35f924880a02f2db7d87a602d2', 'WebQTrn-2904_23a4dc2c184f31defbb00724c9b466b5', 'WebQTest-389_b58a80f073ce333ff592cf88024eb191', 'WebQTrn-3226_bcef0b0a75a7bf2e05f1d95a5459d279', 'WebQTrn-303_904d48695a4599b8d86651a50d4baab6', 'WebQTrn-1597_8adc06ba12a3254370bdb3ff8e8820af', 'WebQTest-1000_3f03848605c6758ff2230a955cd92d65', 'WebQTrn-2871_9e3a08ab28a6799980c632e4dae4f239', 'WebQTrn-962_f69b9d7b30f2ef81193ae7b0a39c78c5', 'WebQTrn-1812_3646d3886548bb03fb73e1545c44afa1', 'WebQTest-1168_f6cf5b0414c7636d8fecd35de820ce08', 'WebQTest-743_a65b27e5670a2d956932182020864234', 'WebQTrn-2653_428127267b489c72416c9298ec80bd7d', 'WebQTrn-2818_7337f6e256ea9b695d810136db74f041', 'WebQTest-106_36ce9951114334bc0a51021ccd3d510a', 'WebQTrn-3057_87afbf2481df60df8476b920f00c4247', 'WebQTrn-3694_73bb87158a068132f7511dc63566c924', 'WebQTrn-2319_5d066fa0939d708dab176e819188e151', 'WebQTest-1923_29c81279ed9a982e12f82e764083db76', 'WebQTest-837_8c90d5499a627a3478313407b1404ecf', 'WebQTrn-2250_98aa1c3bf990bf3d779dee7b611c33fa', 'WebQTrn-1812_af6e94960d6ecddfbb9e8bee3ca654f5', 'WebQTrn-3251_083680ffbc369cb77666e78c0fa3583a', 'WebQTrn-60_ce0d4e864020f94cb6ba262eb64a975e', 'WebQTrn-1577_5d3641e018eabc9011ca0d03911487d3', 'WebQTrn-3251_e7bfd0826590a74bd0a33bd2732849b7', 'WebQTrn-2286_6f8a4ec144197d617a362ba8798b18d0', 'WebQTrn-3033_e00c391d243d235d71cb1b5040e0ae1a', 'WebQTest-1320_2492cfb926882a4bf9c580ba4f7c700a', 'WebQTrn-124_8a391bb9366c22ce0aadc00cfecf7e08', 'WebQTrn-846_3b594835a34214586fc274bdb1d8d0b6', 'WebQTest-1513_a60bea07c81573c12c93e515b219c8b0', 'WebQTest-654_b183c990916cd703ef71720d378b748e', 'WebQTrn-1770_79714129f735fe30ef12c7dc45814f91', 'WebQTest-213_a301052c4da71e75def5ad8b69f9b06c', 'WebQTrn-2069_cf3638075fd9204c82c8adf9ae47925e', 'WebQTest-1686_b30993b3354e5b4bb3ded1e07150aef8', 'WebQTrn-567_d903bf47d5683f31ebe9f50099ecba09', 'WebQTest-1473_4cab2477ddfe14ba3e1ac774680a1f75', 'WebQTrn-2653_aa9cd213dd7b3a0d3e98fdf230b024e3', 'WebQTrn-2754_a52b59230c3401a9e65db7fbf5acf8fa', 'WebQTest-989_1fd20dc9a71887010a0cd60bfd0915c8', 'WebQTrn-3249_75f7d39d822bda23a91efa98ddabf6e8', 'WebQTrn-1394_a7c91a5ecf542ed3feb4615f6d882106', 'WebQTrn-3083_8784ca0ffa5adf55625c8af4f6b657b3', 'WebQTrn-2069_52c8e056ec6add6f53e03ba70667b590', 'WebQTrn-124_c912690c9194dd57e64b6f69c3914a48', 'WebQTrn-436_f7cfe105d2830ecc77ff720612084de8', 'WebQTest-1840_c9b29298640eb6e6b30bb75ceae98297', 'WebQTest-100_a3580acde8969d20b4fe1090a1be6761', 'WebQTrn-1561_4799d8901e0073eaebbc2b893ab30eda', 'WebQTest-1686_97d1854efab5172bdfe763781a23257e', 'WebQTrn-1069_e87644944e029d31491a748a4b4c824b', 'WebQTrn-2784_36f76676900330ffc2a4d3d3d872f99d', 'WebQTest-542_b82441a0a2ccf88afa7fb143e9f5e449', 'WebQTest-1923_c6ade030fef329e1ed98d55f02488885', 'WebQTrn-2316_d579418670907fcb38173073e41bedd1', 'WebQTrn-2152_8f4540de9428b11fdbf25cdc40ff6402', 'WebQTest-1923_44adab808bc1863fab0f07e6556afd67', 'WebQTrn-62_1901065f16bfad63f514f48ecebd767e', 'WebQTrn-64_984678664419049133d2bd748b68df1b', 'WebQTrn-2292_ea0bc3bb340865025c534c91eca19be9', 'WebQTest-1384_ae0ef655e304be3e017fd9cd4955fd62', 'WebQTest-1528_bc90865d2290e81cf10b04c182a63e3b', 'WebQTest-1002_7e0a9b05b43e888a672c7f1d2a9e094d', 'WebQTest-69_365cba904abc355bc084b442a5a4ab37', 'WebQTest-1348_046b26216c66e08d2b528b971292e4e6', 'WebQTrn-25_64dcfc659ae9ffee5e555443282b4e1f', 'WebQTest-1785_8ea20168feb92c6e96c04d0fa86e15a9', 'WebQTrn-1180_2508a12bd20794f8336bf8a4711ae807', 'WebQTrn-1706_7730c9d2fac2a2208c9d9d1e078c5e1b', 'WebQTrn-2047_abc649abeb844da445ed0d372259cd39', 'WebQTest-1797_6e07a94c61aafc2ffd9d6eba802dd88c', 'WebQTrn-3433_675e2a842506fdbbcf3ccdec8f51cee3', 'WebQTrn-3251_020db712f19c955d1c6a004afba5ec2f', 'WebQTrn-21_86bde4a3d89de3a9f4e467f289982d13', 'WebQTrn-2569_382323d8e4417de2b2ee78b7dcf62771', 'WebQTest-1785_b4d5faeb89f3e94de99881cfb0130422', 'WebQTest-538_2b68c4f2a49e647c3da4d7544457a0ce', 'WebQTrn-837_832d3fbdb7319eb9baa01ef8068432cf', 'WebQTest-942_838b888de3d170ac0236da18c8d329e0', 'WebQTrn-2316_4644f7a827779692f5a3f3f894acacd9', 'WebQTrn-846_27a3b20f7d1aebfeea7ca284d80cceb5', 'WebQTrn-2784_3db4e5394a73964976f58989d87836ea', 'WebQTest-1528_f4031bab4675b90da1eff58abc5ecc91', 'WebQTrn-2319_d482e9d52e0f0ee50645cc918c643d6b', 'WebQTest-759_2a6f2d7311c4070f0f98f8526d05907c', 'WebQTest-1797_431f72b2979d844870b6df636d14ac61', 'WebQTest-1923_af65fd059bfafea614fc158d24fe4509', 'WebQTrn-1812_7fa788f49ac13510f448ee288ddd97a0', 'WebQTest-1528_c10f21ff1b8944c429bdb86f46cc9196', 'WebQTrn-894_8c3a4e617ef58ad8f45e3de29a55c96d', 'WebQTrn-2189_839fa68cda37e890bb3eed45714fcc43', 'WebQTrn-3769_6e0a31334a35147f868ae514f9eb9ab8', 'WebQTest-1289_104f814ab1b9df67d3e07b2912c529ae', 'WebQTest-537_e35bbe9c7fc2677853b0df65b7b656f6', 'WebQTrn-513_a73ff21ac29883f58299cd0c2816d5db', 'WebQTest-1785_afeade6c28913208addc4d0685fadb3b', 'WebQTest-538_0c23f06fbbf4e8b9a5c2488c612b2536', 'WebQTrn-452_53d71878d3e6f070410f4d1eaf3b5653', 'WebQTest-1560_a720c55d8c1ac8b92202531c043f6cdd', 'WebQTest-212_a5e3e004e8a3ba709d16682f32e12f2f', 'WebQTrn-567_3fc1da87d6688d1e68a71f1721401694', 'WebQTrn-1677_c592897b4f19683f397c99d4bce6c2d9', 'WebQTrn-241_a60904278de9f6a712bc87d1d25753d2', 'WebQTrn-1180_d234333cfe0037241bacdcbbc5a74317', 'WebQTrn-1677_332868a709543ca2a72cb6c83bb618bb', 'WebQTrn-3249_5aa3063566fca0f5ff73b2b811e3b1ab', 'WebQTest-1797_2fb9e2823ccf35d2103fa8846d6f2ca8', 'WebQTrn-513_cb932de1d6b8743ca91023a47b216435', 'WebQTest-213_61516138603e1aae7ee58e812cc4a2d1', 'WebQTest-1603_6f7bafc08148c28d022ab04b0e6a4aaf', 'WebQTrn-3769_47cc02e7e7f83568581f65b7dd3449dd', 'WebQTrn-2904_b7d507132473eac448dd5f4b20b774f4', 'WebQTest-1560_5f4eeb5e3eb2a683cb6ccec4ddfa7e0e', 'WebQTest-1797_5a1c66f408118c6342c4a0eb350b58de', 'WebQTrn-1665_fd084bd1cf43378d8b0a0648f8215ba4', 'WebQTrn-2092_cafc8c4a401181261ffc3fdc8f490e19', 'WebQTrn-62_b2f9ab8521bc316c4c97cfbba4229932', 'WebQTrn-2904_9e23cfd67cc478f1dd0942be05f97881', 'WebQTrn-2904_3833f4a529eda2b01fd6d8ba457f58b0', 'WebQTest-538_485b6c9f1bd7bd7972f3dd4fcc11b1e9', 'WebQTrn-2974_5e8326632cad769e6a476498d3316697', 'WebQTrn-3249_88bdde5802b7a28527da5deda09a70ab', 'WebQTest-389_0380511448f25dcf175b810016dc9ce5', 'WebQTest-1348_b7598df908bf8cbe941f82e1cefaec28', 'WebQTrn-3249_6dbfa476f0510018f470c29993293e68', 'WebQTrn-3084_d7a8cbddd43733cfb78c3d41af704745', 'WebQTest-1785_4ae1a0927e13f623451a04265c29dc4e', 'WebQTrn-124_771fa831422e1a1f8d7a050a164aedd2', 'WebQTrn-1677_15af244703c8dc90d1e1f3fb2ec8a259', 'WebQTest-1875_a7e655ba2343c8713f26f616c09aa12f', 'WebQTest-450_5839e7f421b1c7dec152856912c23c6f', 'WebQTrn-3543_915d22d3363bc7baa79cb720ec2ec4e6', 'WebQTest-537_ecddf67a2169404aceef47b2109fbe06', 'WebQTrn-2871_363f6eee5c5d62f8931df7cb69917b69', 'WebQTest-918_db8b212b50371e34b8e9349b85059273', 'WebQTrn-25_abd563e9aa3fc7103b5aae5a6d45564a', 'WebQTrn-2784_580feaf976a60d64a0f0308d1310d4e5', 'WebQTest-1035_7908752e63d4926a1c38cc60236acad1', 'WebQTrn-436_d63127b099ed581855690cb5f1524332', 'WebQTrn-567_e4d5f38a9487c935ab4462347b06cf15', 'WebQTest-537_0b9df216de1dc61e927869e6f9dfac2b', 'WebQTest-1528_505f2eda6be64caf7b48a166de833564', 'WebQTrn-2292_c672d27c3d00ea802b18f17142a0c867', 'WebQTrn-846_b2db88607e6e187c67839c81f66e708f', 'WebQTrn-1706_37b57681c5d1e0de5dd91363e9914f93', 'WebQTest-1797_194b220adab0abb76bdf49354cfdf3ff', 'WebQTrn-3226_56c6df98d3f203d856db0584c5dab489', 'WebQTrn-2748_510ee63721c43d65b08c99d91f4f4efc', 'WebQTest-213_cbbd86314870b15371b43439eb40587a', 'WebQTrn-436_b5a2c2761dc6288b6d57aaee16d69bec', 'WebQTrn-2871_98925752e1e60abb73dc775f15ee38af', 'WebQTest-1797_202ded8e5df1c80ea9770154e24e5ffb', 'WebQTrn-14_5b45ce1fa98ff632acb24df13375f4ed', 'WebQTest-759_b9e4b613dd3d82a718a49860d45d6a63', 'WebQTrn-412_e5130693c916cf0420cb76f54b49ddfe', 'WebQTrn-493_722f99e611d343c28690aee2c3f7c640', 'WebQTest-54_549761eca9c4f858651ea00f4f9c4abc', 'WebQTrn-894_dc126021d39290ceeecabd0361b65c4b', 'WebQTrn-846_6463b5d4d186708788ca7b34de172f89', 'WebQTrn-95_a740174dae5d22cf2d93d00318acd471', 'WebQTest-1560_c1c1d3141d16ef82b72cf09954be0346', 'WebQTrn-3249_af94010c3bfc3a584ffb01e553dd2e8f', 'WebQTest-1785_0f8e650edc29ff3b688f4b0f4b86992a', 'WebQTrn-2047_f7b7852c980d3abe73ee87d41804d35d', 'WebQTest-538_c606d81a0b9a8fe9f1a8a17239ebe8e2', 'WebQTest-759_88081dc1a259cbd172b89242384ee1cc', 'WebQTrn-60_067192285b148f91665176a691f7ac79', 'WebQTrn-513_3e812d51c9c00f39fc7165dd0424a78e', 'WebQTest-717_3375a7cecb71d0cf2e2d87d7d513be18', 'WebQTest-1686_87cdd7d2679849d6e7f286145aa6c15f', 'WebQTrn-2316_9f5a12b5581afd827f9fb69956f4af6a', 'WebQTrn-1297_02ed5ce4f0d0e6c1b8d15752a1be80ad', 'WebQTest-212_658c3010a12e95ddde9dbfa9877a7fb7', 'WebQTrn-3251_66da1b47c119995a447078dbc066867f', 'WebQTrn-2047_6d4448f6a30576186670b7bd2be505db', 'WebQTest-1000_1d3a86f839023bb35d439f56a673dbe6', 'WebQTest-626_a7e69ef8a716d6db11c3ba8316c57d9d', 'WebQTrn-1770_bc80eb99b461b02979f7b4186cb73f41', 'WebQTrn-1394_eb44d48fa7660ac55eb6a6894a9d2522', 'WebQTest-12_c34cd4ed0ba2a29e97141049e26b47a2', 'WebQTest-1603_a6d23d999e3bf256d36232e8ff17e675', 'WebQTrn-64_e1d81a60b5954aea1f4417696d8cf733', 'WebQTrn-2653_27bb8ca9da49100394fb007df4a47964', 'WebQTrn-2784_dec6a71a82c29421202060eb63cb1047', 'WebQTest-212_4678a3ae0399fc03eec69ec26c6c9f3f', 'WebQTest-1528_b6c58e02192399b0d08d9b21d16b7c7a', 'WebQTrn-452_5b0f29b4f30e70a790416e24b543f688', 'WebQTrn-3033_46114008e5fa5395b2ea254aaea6985e', 'WebQTrn-2314_2945752f3c92dbdddaf2ae179c5f6e12', 'WebQTrn-3226_1b028c6a2abb6bacf58548f0dbb42ee0', 'WebQTrn-1394_6eb4909a184ab0aac83d44471fb5bd49', 'WebQTrn-2754_7b7ab19b5492a3184356f9305cdd3f9f', 'WebQTest-542_9131e49190aad60dc4fbd564e78cf5d5', 'WebQTrn-2286_219d76ff98f71fe939c88e81e2a66960', 'WebQTrn-2784_7084416ab9f72f1f1f8fc3ce7871ee4a', 'WebQTrn-2540_c5849a4d52a75aa5946a401f89b9b261', 'WebQTest-918_c544bd45cd8f0707031c1ae0dcf1b6f2', 'WebQTrn-2784_7e7ff1a2f6afba2c8cdbaefc5eb6f1af', 'WebQTrn-1938_7322a2a4d46bf36b95bfab4418c9a32b', 'WebQTrn-60_68f0d0ad309d64a4af858a5ef4fb5713', 'WebQTrn-1864_9dc4e22121d3a46d45b8f9bd9e8c7013', 'WebQTrn-3376_0619d288bbed0ca782e60c6f841a6051', 'WebQTest-1686_29e74083744b3631541b29b4094fb273', 'WebQTest-12_68d745a0657c86906382873e57294d6a', 'WebQTrn-2152_3cdf60c15a8355981dd92e3c57ac2eed', 'WebQTrn-1864_67ecd1c247c3b2c9545fbcf1ad8d9d00', 'WebQTrn-557_960c16ffdb29e173df0577fc76c7455d', 'WebQTest-1320_c5498ca807d2e1ec30d4c8fdd41f0bf7', 'WebQTest-100_524908899a8aa334a18a0ac00f8f2fe6', 'WebQTrn-568_d54918e8e89ad97237bce821087a9818', 'WebQTest-538_92e606ef9c0429ad6820797ad2950730', 'WebQTrn-846_a29552911617e890ca2e1d6564e0990e', 'WebQTest-1569_4e73509d14bda62590480b655eee8751', 'WebQTrn-3535_dfcf84e9ba3b9baef4ee1f2cc51994b6', 'WebQTest-996_7c4c41654239ed0a5b99ba5e6481ad5c', 'WebQTrn-1841_cac535ea48922ba42c75ec7ac67cf454', 'WebQTest-1376_b1915c7cd7f743656108611e5095c2e0', 'WebQTest-576_7280a830b2275c6024614d5ba2adae18', 'WebQTest-1686_73c4a5625dc9db03b2331d9937b0fe71', 'WebQTest-12_c701ad2b5b8ef3f3ed26dd2ed8703d05', 'WebQTrn-3100_84e5ab7013a2c0dd11f7f61db45df94c', 'WebQTrn-3335_a9b18d079f29555cbd1e1740c7d6e40e', 'WebQTest-12_7b54b31f3e5a6273f4fd2a20e565ec6d', 'WebQTest-1923_53fa50559ec52976eb1f02fc6bd5ae76', 'WebQTrn-3694_4089d6d7ba86121ff285b80865682a7c', 'WebQTrn-2738_910f8908e2d53dfb850b7b52b74e4a51', 'WebQTest-361_89e48a343537ccd0148b822f63519c99', 'WebQTest-1560_fd6688bbf30871641ad813fcf09d673c', 'WebQTest-361_5f8c05737061d5501b6a23a978b5589b', 'WebQTrn-846_581ac098af1edbcc3d96a126843bb703', 'WebQTrn-2152_b9bbe53d5eba5d74535f24c0afbb82ba', 'WebQTest-1348_d50300a04640da2d3013e859ecdc9257', 'WebQTest-100_1619341addc048f21b246e33e2458609', 'WebQTest-450_2ed540621cbcd0bcda8774699532cf98', 'WebQTrn-2904_cb8d941f825c5dd7b2d6f3faef7f9229', 'WebQTrn-1001_1134a24400661a6f5feeb3b325cbb1df', 'WebQTest-1686_554811ebe1463287ee640a214683ea57', 'WebQTrn-3694_e185ea716e0dcbdf8d9ad6498b5e88f1', 'WebQTrn-2314_f53b7962c4dfca88044f3c0a89ac0290', 'WebQTest-1923_5584d7ea5e4b9f2391d1610bcc5f75fb', 'WebQTest-996_3b05c852523638229a0ba552f2ac27dd', 'WebQTrn-2664_0f1d384919bac2772e7f384ffd0116fe', 'WebQTrn-2576_975ba342df8c19b7e9d8f452f606a994', 'WebQTest-1120_c78405f6f2157645f1098f8a871f1bc2', 'WebQTrn-2576_280f6fb1deae2956230d33d29b4054ae', 'WebQTest-1528_a2694b95163afd08c03e36c6c195cab6', 'WebQTrn-2069_44d67b357829d662abe1c6dc637e219d', 'WebQTrn-372_236e29382c53f33ae0b8f011963256de', 'WebQTrn-2152_b7b2e2a812931135ab913a8613e57c5c', 'WebQTest-213_1bb5d148e28985730d4603ae86c12405', 'WebQTrn-2784_2c591595db3d76bb1d0aaa4cb7411b7a', 'WebQTrn-436_d4fb347b7326355a4457aa2ce1d502f3', 'WebQTest-534_0fca6c57b860f8b3ab53d092652cb048', 'WebQTest-996_e0a199dde06fb3ce678e842ac26c887b', 'WebQTest-212_d5309c79dc99d1d829805d88dd254553', 'WebQTest-590_e1cd6a19c1fe109a00e4149625e31531', 'WebQTrn-513_45edfb1ca18b998b529351260be38b7f', 'WebQTrn-683_3cfa649a5d5692f81e227ca73f32702f', 'WebQTrn-3527_1661062ba428b2ec2a00e2f84257682f', 'WebQTrn-105_839fa68cda37e890bb3eed45714fcc43', 'WebQTest-450_32407709a2c3bb1902f55d63756250a4', 'WebQTrn-2784_6ab5c32d090e32dca9db642f612076b8', 'WebQTrn-3084_73a0a036677106856ef62808aa205b70', 'WebQTest-537_a8a56da657efbba90881cce421ae6962', 'WebQTest-654_6ca8cba7511811830a04cd64a8a4cf77', 'WebQTest-1348_67d901eb5314fe0563f6e2602b72f478', 'WebQTrn-837_6e82d9e61c6b514331784d0cedb1293d', 'WebQTest-538_e650705713c2b0f407612c8935722c71', 'WebQTrn-1770_6a7c160ace84e7908302f805739ad06d', 'WebQTest-106_8a7ae3422867fd4b2d40b06930915468', 'WebQTest-1528_ac372aaf6e68f8a1bd4b4c8e75972875', 'WebQTrn-738_75d0bf8a1c2aca91051597cd3bdfb371', 'WebQTrn-2400_400ed71d115f080056e3437113c51ce2', 'WebQTrn-2026_0c8bcc717d50c92c31ff802641504b43', 'WebQTest-759_d5838dc0b2134997e3ba9eb8dc264db4', 'WebQTrn-2218_05b93d057f56c7f623f0bed078a219b3', 'WebQTrn-894_b3a9eef2f5d4e40fd4221db45d001fd0', 'WebQTrn-3136_9204c006e1ff89d48f18c8dee828333f', 'WebQTrn-567_df97b91c1a9dfe15bdad689feb59f791', 'WebQTrn-1706_673b46863f1573ab93c79cbd80124990', 'WebQTrn-2721_963759f4184e1a7f2fdbeea6f16e017e', 'WebQTest-1965_b7c82cd0420f9f1934037e1625db4685', 'WebQTest-537_a52d362d7abdae05ae86743bdb72a808', 'WebQTest-537_3f765ae8a25eecea63667cdb84f5500a', 'WebQTrn-1629_78cb2280db26c05fdc9b59622dc04611', 'WebQTest-1923_a64ef0f5ce397a5e1ef6fcd550ebfcfb', 'WebQTest-1560_597bd7b2c629237d470d67390eba7edb', 'WebQTrn-837_b5a16676e03dc3374f767914e2292443', 'WebQTest-538_dfc99f6e8277309985f191b4575cee66', 'WebQTest-918_a6b0e48e01f77546d20952acdc67b396', 'WebQTest-1705_35c94dc6944d964963b44949d8ed512c', 'WebQTrn-2871_4c5dc1332d9eba8fde58e7b9561d4ae3', 'WebQTrn-2784_1afdc65142a646a69eb7f01dce690c4b', 'WebQTrn-241_b96b735d1d754f27bc8986b614669f7b', 'WebQTrn-2570_719a19b4c8ab7d1b27c9529c394743ef', 'WebQTest-1686_9f2fd1f7d2cee3d36d73ff5e3ef8b2c1', 'WebQTest-1603_5aabdab4c50d62388224ccc564a50535', 'WebQTrn-2540_58a5d284cc8e461c26e8c88444fd56d4', 'WebQTest-1560_2d851e09c04c135c6499a8e7bcf24ee9', 'WebQTrn-2809_8deb69518817ab56417b04d421af85b3', 'WebQTest-1002_18b698a0c4db4e35a9c98877fc9c40ae', 'WebQTrn-124_5e89e91ab1c8fd1eea3f415d52cb4738', 'WebQTest-1965_b3a2ccc81ab934c7f836221d51e151f5', 'WebQTrn-124_0ebb5db67d331752adddbe06bca557fc', 'WebQTrn-124_405a78f132e69f4771aff40d07cd0294', 'WebQTrn-1677_a3f3297c1fb33fa36090e79b89b6690c', 'WebQTrn-3744_1bc38eddb1fb4d93010f232f991f8d09', 'WebQTrn-1392_ba1b3d8965af4c1f1127fa5219d1c2df', 'WebQTest-1528_f8024d5f486967b395064aa07aa52d6e', 'WebQTrn-2258_a37f457dd3fdc936b7b86a218e415de7', 'WebQTrn-1812_93d2e88cddc04bc55b23c31bab096e16', 'WebQTrn-3376_118a81ac3fc08ca8590bf8be836d1be1', 'WebQTrn-3412_305f49b65a760af0e6ce8d06f43aa22a', 'WebQTrn-1677_87deda135e48fdbc80ef86a95018f5c9', 'WebQTest-538_e92ac6479096acfc8a23fa7ff3fa1b37', 'WebQTrn-1731_843303055781fd274c0e177ae95bbb5a', 'WebQTrn-124_24e0665d6867419f9f3d2e4a2c1a997f', 'WebQTrn-1646_0c8bcc717d50c92c31ff802641504b43', 'WebQTrn-3400_675e2a842506fdbbcf3ccdec8f51cee3', 'WebQTrn-3249_a5b4b068155de58edc7e632f9bece371', 'WebQTrn-3384_8728d31e67df2c6d6395deed585552cc', 'WebQTest-213_7f3bd401a2fe034b67eb41db6a3801b2', 'WebQTrn-2576_49a64967e14914f15dcaf15424cd6f9b', 'WebQTest-12_64c67292548a2872e94c2d2162850e82', 'WebQTrn-1294_5824e654d0c53e3c1bcda804098340f5', 'WebQTrn-1812_19c95ace02f0dd9b7f67de3ecc3e8f5a', 'WebQTrn-2784_06d1c376b588c7a7bb36e30c95e914e7', 'WebQTrn-3766_eaeb39a0db717338715a044a8175bd9a', 'WebQTrn-3249_c2f7c4d755631300e1e557415c49ebe1', 'WebQTrn-497_e0a9871be5d3b47874f72ae67dbbc458', 'WebQTrn-1938_501c6712d98f6b8b1f2db4599112f057', 'WebQTrn-1812_8126be584feae3f358a930823db22c4e', 'WebQTest-576_e9a3bdf6f62a7b6057f7d339fb48c28f', 'WebQTest-1560_9b121a33fc0ee97ee3a3c43ee4e02a78', 'WebQTest-1923_3c662be721b9c2107f0dd05ba38cf729', 'WebQTest-1965_de1aeafb354b0cc40a64b7579efade3e', 'WebQTrn-1677_6ba2b86bf1f3e713c356c221e346eb5e', 'WebQTest-537_38a32ea03bb0de4387121b28a883a92e', 'WebQTrn-2834_61efa0b17cc53c2187422d88bf670f22', 'WebQTrn-1677_b3db2e89d0cb9edd4d5e69ebe1ba5608', 'WebQTrn-3251_acf68a88c0375f50017818448cac05ac']

    
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
        # pdb.set_trace()
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
        args.predict_path, args.d, 'neg_cwq', args.split, "seperate3"
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

    output_file = os.path.join(output_dir, f"FullRoggen17_beam_predictions.jsonl")
    # pdb.set_trace()
    fout, processed_list = get_output_file(output_file, force=args.force)
    # pdb.set_trace()
    dataset = Dataset.from_dict(dataset[3000:])
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
            # if data['id'] in without_path_list or data['id'] in bad_qid:
            #     continue
            if len(data['predicted_paths']) == 0:
                print('pass no rule', data['id'])
                continue
            if data['id'] in badid_list:
                print('pass bad id', data['id'])
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
        "--data_path", type=str, default="xxxxxx"
    )
    argparser.add_argument("--d", "-d", type=str, default="webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="xxxxxxx")
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
        default="xxxxxx",
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
        default="xxxxx",
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




    