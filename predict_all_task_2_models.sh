#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip fr dev data/predictions-camembert-camembert-large-fr-1-task2 "models/camembert-camembert-large-fr-1-task2/final/" 2
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip it dev data/predictions-dbmdz-bert-base-italian-xxl-uncased-it-1-task2 "models/dbmdz-bert-base-italian-xxl-uncased-it-1-task2/final/" 2
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip de dev data/predictions-german-nlp-group-electra-base-german-uncased-de-1-task2 "models/german-nlp-group-electra-base-german-uncased-de-1-task2/final/" 2
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip en dev data/predictions-roberta-large-en-1-task2 "models/roberta-large-en-1-task2/final/" 2
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip en dev data/predictions-xlm-roberta-large-en-1-task2 "models/3epochmodels/xlm-roberta-large-en-1-task2/final/" 2

python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip de dev data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip fr dev data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip it dev data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip en dev data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2


python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip de test data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip fr test data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip it test data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip en test data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2

python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip de surprise_test data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip fr surprise_test data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip it surprise_test data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip en surprise_test data/predictions-final-multilang-task2 "models/3epochmodels/xlm-roberta-large-en-de-fr-it-1-task2/final/" 2
