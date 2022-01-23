python run_mlm.py \
    --run_name lm-model-512-batch-full \
    --model_name_or_path dbmdz/bert-base-german-uncased \
    --train_file data/sepp_nlg_train_de.txt \
    --validation_file data/sepp_nlg_dev_de.txt \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --preprocessing_num_workers 8 \
    --logging_steps 10 \
    --line_by_line True \
    --num_train_epochs 3 \
    --learning_rate 4e-5 \
    --gradient_accumulation_steps 64 \
    --save_total_limit 3 \
    --fp16 True \
    --output_dir models/sepp2021-de-512-full

#    --evaluation_strategy epoch \
# total tran sampels 1426351
#--max_train_samples 500000 \
#--max_val_samples 500000 \
# use validation_split_percentage
#--tpu_num_cores

