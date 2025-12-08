MODEL_PATH=/workspace/longxiao/LLM/Llama-2-7b-chat-hf
# DATASET_LIST="/workspace/longxiao/KGQA/EPERM/datasets/joint_training/align/cwq/cwq_train.jsonl 
#               /workspace/longxiao/KGQA/EPERM/datasets/joint_training/align/webqsp/webqsp_train.jsonl 
#               /workspace/longxiao/KGQA/EPERM/datasets/joint_training/qa/webqsp/webqsp_train.jsonl 
#               /workspace/longxiao/KGQA/EPERM/datasets/joint_training/qa/cwq/cwq_train.jsonl 
#               /workspace/longxiao/KGQA/EPERM/datasets/joint_training/ExplainQAData/cwq/cwq_train_1000.jsonl 
#               /workspace/longxiao/KGQA/EPERM/datasets/joint_training/ExplainQAData/webqsp/webqsp_train_1000.jsonl"

DATASET_LIST="/workspace/longxiao/KGQA/EPERM/datasets/joint_training/align/cwq/cwq_train.jsonl 
              /workspace/longxiao/KGQA/EPERM/datasets/joint_training/ExplainQAData/cwq/cwq_train_1000.jsonl
              /workspace/longxiao/KGQA/EPERM/datasets/joint_training/qa/cwq/cwq_train_with_neg.jsonl
              /workspace/longxiao/KGQA/EPERM/datasets/joint_training/align/webqsp/webqsp_train.jsonl
              /workspace/longxiao/KGQA/EPERM/datasets/joint_training/qa/webqsp/webqsp_train.jsonl
              /workspace/longxiao/KGQA/EPERM/datasets/joint_training/ExplainQAData/webqsp/webqsp_train_1000.jsonl"
              

# DATASET_LIST="/workspace/longxiao/KGQA/EPERM/datasets/joint_training/align/cwq/cwq_train_3hops_filter.jsonl
#               /workspace/longxiao/KGQA/EPERM/datasets/joint_training/align/webqsp/webqsp_train_3hops_filter.jsonl
#               /workspace/longxiao/KGQA/EPERM/datasets/joint_training/qa/cwq/cwq_train_3hops_filter.jsonl
#               /workspace/longxiao/KGQA/EPERM/datasets/joint_training/qa/webqsp/webqsp_train_3hops_filter.jsonl"
# /workspace/longxiao/KGQA/EPERM/datasets/joint_training/qa/cwq/cwq_train.jsonl 

# DATASET_LIST="/workspace/longxiao/KGQA/EPERM/datasets/joint_training/align/cwq/cwq_train.jsonl"
SAVE_NAME=EPERM_allnegcwq
SAVE_PATH=/workspace/longxiao/KGQA/EPERM/save_models/${SAVE_NAME}
ADD_REL=False 

accelerate launch --config_file /workspace/longxiao/KGQA/EPERM/config/deepspeed_zero3.yml /workspace/longxiao/KGQA/EPERM/src/joint_training/joint_finetuning.py \
    --data_path_list ${DATASET_LIST}  \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --add_rel_token ${ADD_REL} \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --run_name ${SAVE_NAME}
