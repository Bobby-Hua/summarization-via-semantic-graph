DATA_DIR=FOLDER_OF_YOUR_CHOICE
AMR_DATA_DIR=FOLDER_OF_YOUR_CHOICE
AMR_VOCAB=FOLDER_OF_YOUR_CHOICE
OUT_DIR=FOLDER_OF_YOUR_CHOICE
CACHE_DIR=FOLDER_OF_YOUR_CHOICE

TRAIN_AMR_FILE=${AMR_DATA_DIR}/AMI_TRAIN_amr_input.json
EVAL_AMR_FILE=${AMR_DATA_DIR}/AMI_VAL_amr_input.json
TEST_AMR_FILE=${AMR_DATA_DIR}/AMI_TEST_amr_input.json

train_ep_to_scene=${AMR_DATA_DIR}/AMI_TRAIN_ep_to_scene_scene_merged.json
eval_ep_to_scene=${AMR_DATA_DIR}/AMI_VAL_ep_to_scene_scene_merged.json
test_ep_to_scene=${AMR_DATA_DIR}/AMI_TEST_ep_to_scene_scene_merged.json
TARGET_CONCEPT_FILE=${AMR_DATA_DIR}/ami_train_target_concept_ep.pickle

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

python run_summarization_amr.py \
    --model_name_or_path MingZhong/DialogLED-base-16384 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy steps \
    --train_file ${DATA_DIR}/train.src.json \
    --validation_file ${DATA_DIR}/val.src.json \
    --test_file ${DATA_DIR}/test.src.json \
    --text_column src \
    --summary_column tgt \
    --output_dir ${OUT_DIR} \
    --label_smoothing_factor 0.1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_source_length 8192 \
    --max_target_length 360 \
    --val_max_target_length 360 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --max_steps 400 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --save_steps 10 \
    --eval_steps 10 \
    --predict_with_generate \
    --train_amr_file ${TRAIN_AMR_FILE} \
    --eval_amr_file ${EVAL_AMR_FILE} \
    --test_amr_file ${TEST_AMR_FILE} \
    --token_vocab ${AMR_VOCAB}/token_vocab \
    --concept_vocab ${AMR_VOCAB}/concept_vocab \
    --predictable_token_vocab ${AMR_VOCAB}/predictable_token_vocab \
    --relation_vocab ${AMR_VOCAB}/relation_vocab \
    --token_char_vocab ${AMR_VOCAB}/token_char_vocab \
    --concept_char_vocab ${AMR_VOCAB}/concept_char_vocab \
    --train_ep_to_scene ${train_ep_to_scene} \
    --eval_ep_to_scene ${eval_ep_to_scene} \
    --test_ep_to_scene ${test_ep_to_scene} \
    --cached_features_dir ${CACHE_DIR} \
    --fp16 \
    --target_concept ${TARGET_CONCEPT_FILE}