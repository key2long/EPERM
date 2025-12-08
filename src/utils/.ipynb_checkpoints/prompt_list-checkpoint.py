score_entity_candidates_prompt = """[INST] <<SYS>>\n<</SYS>>
I want to answer the question through a relationship path. There will be multiple candidate entities along the path. Please help me choose the entity that can better infer the answer to the question.

#EXAMPLE#
Question: who did cristiano ronaldo play for in 2010?
Paths: soccer.football_player_match_participation.player, soccer.football_player_match_participation.team
Candidate Entity: m.0g9lr08
Relevent Information: 
m.0g9lr08 --> soccer.football_player_match_participation.match --> 2010 FIFA World Cup Group G - POR ./. PRK
m.0g9lr08 --> soccer.football_player_match_participation.team --> Portugal national football team
m.0g9lr08 --> soccer.football_player_match_participation.shirt_number --> 7
m.0g9lr08 --> soccer.football_player_match_participation.part_of_starting_lineup --> true \n

Inference criteria: To score the contribution of entities to the question, we need to determine if the entities are relevant to the question based on provided relevant information. For example, 
if the relevant information of entity m.0g9lr08 includes the 2010 FIFA World Cup, which is consistent with the time information in the question, there is a higher likelihood 
that this entity is relevant to the question. If the relevant information of an entity is not consistent with the time information "2010" in the question, the relevance between 
that entity and the question is lower, and the score will be lower such as 0.56. Based on the above speculative evidence, the scores of m.0g9lr08 could be assigned 0.93(high relevance).



#INPUT#
Question:{}
Paths:{}
Candidate Entity:{}
Relevent Information:{}
Please help me score the relevance between the Candidate Entity {} and the question based on the relevent information of the Candidate Entity in the #INPUT#. The scoring range is from 0 to 1 and different entity has different socre. Finally, please output the different socre in the following format: {{Candidate Entity:{},Score:xxx}}, and do not output explanations.
[/INST]
"""

# Output: {{Candidate Entity:m.0g9lr08,Score:0.93}}
# score_entity_candidates_prompt = """[INST] <<SYS>>\n<</SYS>>
# #CONTEXT#
# I want to answer the question through a relationship path. There will be multiple candidate entities along the path. Please help me choose the entity that can better infer the answer to the question.

# #EXAMPLE#
# Question: who did cristiano ronaldo play for in 2010?

# Paths: soccer.football_player_match_participation.player, soccer.football_player_match_participation.team

# Candidate Entity: m.0g9lr08

# Relevent Information: 
# m.0g9lr08 --> soccer.football_player_match_participation.match --> 2010 FIFA World Cup Group G - POR ./. PRK
# m.0g9lr08 --> soccer.football_player_match_participation.team --> Portugal national football team
# m.0g9lr08 --> soccer.football_player_match_participation.shirt_number --> 7
# m.0g9lr08 --> soccer.football_player_match_participation.part_of_starting_lineup --> true \n

# Inference criteria: To score the contribution of entities to the question, we need to determine if the entities are relevant to the question based on provided relevant information. For example, 
# if the relevant information of entity m.0g9lr08 includes the 2010 FIFA World Cup, which is consistent with the time information in the question, there is a higher likelihood 
# that this entity is relevant to the question. If the relevant information of an entity is not consistent with the time information "2010" in the question, the relevance between 
# that entity and the question is lower, and the score will be lower as well. Based on the above speculative evidence, the scores of m.0g9lr08 could be assigned 0.93.

# Output: {{Candidate Entity:m.0g9lr08,Score:0.93}}

# #INPUT#
# Question:{}
# Paths:{}
# Candidate Entity:{}
# Relevent Information:{}

# #RESPONSE#
# Please help me score the relevance between the Candidate Entity {} and the question based on the relevent information of the Candidate Entity in the #INPUT#. The scoring range is from 0 to 1. Finally, please output the results in the following format: {{Candidate Entity:{},Score:xxx}}, and do not output explanations.
# [/INST]
# """

# Please help me score the contributions of the candidate entity in the following inference path to the question based on their relevant information. On a scale of 0 to 1.
# Please output the result in the following format: {{Entity ID: xxx, Score: xxx}}
#RESPONSE#
# Please score the contributions of the candidate entity which in the Entity ID in the following inference path to the question based on their relevant information. On a scale of 0 to 1. Finally, please output the results in the following format: {{Entity_ID:xxx,Score:xxx}}, and do not output the explanation.
# Please score the entity ID based on its relevant information in the following inference path. The scoring range is from 0 to 1. Finally, please output the results in the following format: {{Entity_ID:xxx,Score:xxx}}, and do not output explanations.
# Please score the entity ID based on its relevant information in the following inference path. The scoring range is from 0 to 1. Finally, please output the results in the following format: {{Entity_ID:xxx,Score:xxx}}, and do not output explanations.

# reasoning_prompt = """[INST] <<SYS>>\n<</SYS>>
# #CONTEXT#
# I want to reason the answer to the question through a series of paths, each path has a confidence score for reasoning the question. Please infer the answer to the question based on these paths and their corresponding scores.

# #OBJECTIVE#
# Based on the reasoning paths and their confidence scores, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.

# # RESPONSE #
# Please output the prediction entity in the following format: entity1, entity2, ...


# #OBJECTIVE#
# Please help me score the relevance between the candidate entity and the question based on the relevent information of the related entities connected to the candidate entity. The scoring range is from 0 to 1.

# # RESPONSE #
# Please output the score in the following format: {{Candidate Entity: xxx, Score: xxx}}


# \n\nReasoning Paths and Scores:{}
# \n\nQuestion:\n{}
# [/INST]
# """

####################################################
reasoning_prompt = """[INST] <<SYS>>\n<</SYS>>
Please answer the question based on the scores of the reasoning paths and return all possible answers in a list. Each path has a score (0.0-1.0) at its end, and please strictly adhere to the prediction criterion of outputting the entity with a high score.
\nReasoning Paths and their scores:\n{}
\nQuestion:\n{}
[/INST]
"""
########################################################

# #EXAMPLE#
# Please answer the question based on the scores of the reasoning paths and return all possible answers in a list. Each path has a score (0.0-1.0) at its end, and please output the entity in the end of the path with the highest score as the prediction.
# Hello, please answer the question based on the confidence scores of the reasoning paths. Each path has a confidence score (0-1) at its end, please output the entity at the end of the path by the highest confidence score as the prediction. Please answer the question as concisely as possible and return all possible answers in a list. 
# Hello, please answer the question based on the reasoning paths and their confidence scores. Entities corresponding to higher path scores are more likely to be the answers. Please keep the answer as simple as possible and return all possible answers as a list.

# Question: who was the original voice of C on movieD?

# Reasoning Paths and Scores:
# \nC -> tv.tv_character.appeared_in_tv_episodes -> m.09p1747 -> tv.tv_actor.guest_roles -> A\tScore:0.302
# \nC -> tv.tv_character.appeared_in_tv_episodes -> m.0jzvxtw -> tv.tv_actor.guest_roles -> A\tScore:0.292
# \nC -> tv.tv_character.appeared_in_tv_episodes -> g.11byb39pmc -> tv.tv_actor.guest_roles -> A\tScore:0.322
# \nC -> tv.regular_tv_appearance.character -> m.04d4q86 -> tv.tv_actor.starring_roles -> B\tScore:0.322
# \nC -> tv.regular_tv_appearance.character -> m.0k6pxpv -> tv.tv_actor.starring_roles -> B\tScore:0.312

# output: B

# Inference criteria: 
# In all reasoning paths, the score for the path C -> tv.regular_tv_appearance.character -> m.04d4q86 -> tv.tv_actor.starring_roles -> B is 0.322 and it is the highest score, 
# which indicates a higher probability to infer that the final result is Lacey Chabert. If there are some paths with similar and relatively high scores, then the results B derived from those paths could all possibly be correct results.

# reasoning_prompt = """[INST] <<SYS>>\n<</SYS>>
# Based on the reasoning paths and their confidence scores to question answering, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.

# #EXAMPLE#
# Question: who was the original voice of meg griffin on family guy?

# Reasoning Paths and Scores:
# \nMeg Griffin -> tv.tv_character.appeared_in_tv_episodes -> m.09p1747 -> tv.tv_actor.guest_roles -> Mila Kunis\tScore:0.302
# \nMeg Griffin -> tv.tv_character.appeared_in_tv_episodes -> m.0jzvxtw -> tv.tv_actor.guest_roles -> Mila Kunis\tScore:0.292
# \nMeg Griffin -> tv.tv_character.appeared_in_tv_episodes -> g.11byb39pmc -> tv.tv_actor.guest_roles -> Mila Kunis\tScore:0.322
# \nMeg Griffin -> tv.regular_tv_appearance.character -> m.04d4q86 -> tv.tv_actor.starring_roles -> Lacey Chabert\tScore:0.322
# \nMeg Griffin -> tv.regular_tv_appearance.character -> m.0k6pxpv -> tv.tv_actor.starring_roles -> Lacey Chabert\tScore:0.312

# output: Lacey Chabert

# Inference criteria: 
# In all reasoning paths, the score for the path Meg Griffin -> tv.regular_tv_appearance.character -> m.04d4q86 -> tv.tv_actor.starring_roles -> Lacey Chabert is 0.322 and it is the highest score, 
# which indicates a higher probability to infer that the final result is Lacey Chabert. If there are some paths with similar and relatively high scores, then the results derived from those paths could all possibly be correct results.


######################## original #########################################################
# reasoning_prompt = """[INST] <<SYS>>\n<</SYS>>
# Based on the reasoning paths and their confidence scores to question answering, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.
# \nReasoning Paths:\n{}
# \nQuestion:\n{}
# [/INST]
# """