#python predict_transformer.py data/sepp_nlg_2021_train_dev_data.zip de dev data/predictions-xlm-roberta-de "models/xlm-roberta-base-task1-multi 2021-03-24 22:21:05/final"
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data.zip en dev data/predictions-xlm-roberta-en "models/xlm-roberta-base-task1-multi 2021-03-24 22:21:05/final"
python predict_transformer.py data/sepp_nlg_2021_train_dev_data.zip fr dev data/predictions-xlm-roberta-fr "models/xlm-roberta-base-task1-multi 2021-03-24 22:21:05/final"
python predict_transformer.py data/sepp_nlg_2021_train_dev_data.zip it dev data/predictions-xlm-roberta-it "models/xlm-roberta-base-task1-multi 2021-03-24 22:21:05/final"

python evaluate_sepp_nlg_2021_subtask1.py data/sepp_nlg_2021_train_dev_data.zip fr dev data/predictions-xlm-roberta-fr
python evaluate_sepp_nlg_2021_subtask1.py data/sepp_nlg_2021_train_dev_data.zip it dev data/predictions-xlm-roberta-it
#python evaluate_sepp_nlg_2021_subtask1.py data/sepp_nlg_2021_train_dev_data.zip en dev data/predictions-xlm-roberta-en
#python evaluate_sepp_nlg_2021_subtask1.py data/sepp_nlg_2021_train_dev_data.zip de dev data/predictions-xlm-roberta-de