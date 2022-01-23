
#echo "dev"
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip fr dev data/predictions-camembert-camembert-large-fr-1 "models/camembert-camembert-large-fr-1/final/" 1
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip it dev data/predictions-dbmdz-bert-base-italian-xxl-uncased-it-1 "models/dbmdz-bert-base-italian-xxl-uncased-it-1/final/" 1
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip de dev data/predictions-german-nlp-group-electra-base-german-uncased-de-1 "models/german-nlp-group-electra-base-german-uncased-de-1/final/" 1
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip en dev data/predictions-roberta-large-en-1 "models/roberta-large-en-1/final/" 1
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip en dev data/predictions-xlm-roberta-large-en-1 "models/xlm-roberta-large-en-1/final/" 1
#python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip de dev data/predictions-german-nlp-group-electra-base-german-uncased-de-1-task1 "models/german-nlp-group-electra-base-german-uncased-de-1-task1/final/" 1


python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip de dev data/predictions-final-german-task1 "models/3epochmodels/german-nlp-group-electra-base-german-uncased-de-1-task1-da-full/final/" 1
python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip de dev data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip fr dev data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip it dev data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_train_dev_data_v5.zip en dev data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1


echo "test"
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip de test data/predictions-final-german-task1 "models/3epochmodels/german-nlp-group-electra-base-german-uncased-de-1-task1-da-full/final/" 1
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip de test data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip fr test data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip it test data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip en test data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1

echo "surprise test"
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip de surprise_test data/predictions-final-german-task1 "models/3epochmodels/german-nlp-group-electra-base-german-uncased-de-1-task1-da-full/final/" 1
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip de surprise_test data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip fr surprise_test data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip it surprise_test data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1
python predict_transformer.py data/sepp_nlg_2021_test_data_unlabeled_v5.zip en surprise_test data/predictions-final-multilang-task1 "models/xlm-roberta-large-en-de-fr-it-1-task1/final/" 1

       



