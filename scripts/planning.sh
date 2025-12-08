SPLIT="train"
DATASET_LIST="cwq"
MODEL_NAME=EPERM_cwq
# MODEL_PATH='/workspace/longxiao/KGQA/EPERM/scripts/save_models/EPERM'
# MODEL_PATH='/workspace/longxiao/KGQA/EPERM/save_models/EPERM_genpath'
MODEL_PATH='/workspace/longxiao/KGQA/EPERM/save_models/EPERM_cwq'
# MODEL_PATH='/workspace/longxiao/KGQA/EPERM/save_models/EPERM_genpath_withfour'

BEAM_LIST="6" # "1 2 3 4 5"
for DATASET in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        python /workspace/longxiao/KGQA/EPERM/src/qa_prediction/gen_rule_path.py \
        --model_name ${MODEL_NAME} \
        --model_path ${MODEL_PATH} \
        -d ${DATASET} \
        --split ${SPLIT} \
        --n_beam ${N_BEAM}
    done
done

