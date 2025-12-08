SPLIT="test"
DATASET_LIST="webqsp"
MODEL_NAME=EPERM_genpath
PROMPT_PATH=/workspace/longxiao/KGQA/EPERM/reasoning-on-graphs-master/prompts/llama2_predict.txt
BEAM_LIST="8" # "1 2 3 4 5"

for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        RULE_PATH=/workspace/longxiao/KGQA/EPERM/reasoning-on-graphs-master/results/gen_rule_path/webqsp/EPERM_genpath/predictions_inter2_8_6_withscore_False.jsonl
        python /workspace/longxiao/KGQA/EPERM/reasoning-on-graphs-master/src/qa_prediction/sort_predict_answer.py \
            --model_name ${MODEL_NAME} \
            -d ${DATA_NAME} \
            --prompt_path ${PROMPT_PATH} \
            --add_rule \
            --rule_path ${RULE_PATH} \
            --model_path '/workspace/longxiao/KGQA/EPERM/reasoning-on-graphs-master/save_models/EPERM_genpath'
    done
done
# /workspace/longxiao/KGQA/EPERM/reasoning-on-graphs-master/save_models/EPERM_genpath
# /workspace/longxiao/KGQA/EPERM/reasoning-on-graphs-master/results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
# --model_path '/workspace/longxiao/KGQA/EPERM/reasoning-on-graphs-master/scripts/save_models/EPERM' \
            # --model_path '/workspace/longxiao/LLM/Llama-2-7b-chat-hf'\