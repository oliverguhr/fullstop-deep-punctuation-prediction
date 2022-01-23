#python predict_transformer.py data/sepp_nlg_2021_train_dev_data.zip de dev data/predictions-roberta-large-de "models/roberta-large-task1-multi 2021-03-25 13:12:06/final"
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data.zip fr dev data/predictions-roberta-large-fr "models/roberta-large-task1-multi 2021-03-25 13:12:06/final"
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data.zip en dev data/predictions-roberta-large-en "models/roberta-large-task1-multi 2021-03-25 13:12:06/final"
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data.zip it dev data/predictions-roberta-large-it "models/roberta-large-task1-multi 2021-03-25 13:12:06/final"

python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip de dev data/predictions-test "models/german-nlp-group-electra-base-german-uncased-2-adamw-optimal-hyperparameter-v5data 2021-05-04 18:40:51/final" 2
python evaluate_sepp_nlg_2021_subtask2.py data/sepp_nlg_2021_train_dev_data_v5.zip de dev data/predictions-test

#python evaluate_sepp_nlg_2021_subtask1.py data/sepp_nlg_2021_train_dev_data.zip fr dev data/predictions-roberta-large-fr
#python evaluate_sepp_nlg_2021_subtask1.py data/sepp_nlg_2021_train_dev_data.zip it dev data/predictions-roberta-large-it
#python evaluate_sepp_nlg_2021_subtask1.py data/sepp_nlg_2021_train_dev_data.zip en dev data/predictions-roberta-large-en
#python evaluate_sepp_nlg_2021_subtask1.py data/sepp_nlg_2021_train_dev_data.zip de dev data/predictions-roberta-large-de