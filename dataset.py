import re
import io
import os
from zipfile import ZipFile
from tqdm import tqdm
import pickle
import random

def map_task_two_label(label):
    if  label not in [':', '?', '-', ',', '.']: #label != "0" and label != "." and label != "," and label != "?" and label != "!" and label != ";" and :
        return "0"
    return label

def load_from_zip(data_zip: str, data_set: str, lang: str, subtask: int = 1):
    """
    Loads every file from the dataset into an array.
    Subtask is either 1 or 2.
    """
    if data_set == "aug":
        relevant_dir = "" # all files are relevant..
    else:
        relevant_dir = os.path.join('sepp_nlg_2021_data', lang, data_set)        
        relevant_dir = re.sub(r'\\', '/', relevant_dir)

    all_gt_labels, all_predicted_labels = list(), list()  # aggregate all labels over all files
    with ZipFile(data_zip, 'r') as zf:  # load ground truth
        fnames = zf.namelist()
        gt_tsv_files = [
            fname for fname in fnames
            if fname.startswith(relevant_dir) and fname.endswith('.tsv')
        ]        

        data = []
        for i, gt_tsv_file in enumerate(tqdm(gt_tsv_files), 1):
            #print(i, gt_tsv_file)
            basename = os.path.basename(gt_tsv_file)

            # get ground truth labels
            with io.TextIOWrapper(zf.open(gt_tsv_file), encoding="utf-8") as f:
                lines = f.read().strip().split('\n')
                rows = [line.split('\t') for line in lines]
                words = [row[0] for row in rows]                
                if subtask == 1:
                    labels = [row[subtask] for row in rows]
                else:
                    labels = [map_task_two_label(row[subtask]) for row in rows]
                if len(words) != len(labels):
                    raise Exception( "word / label missmatch in file " + gt_tsv_file)
                data.append([words,labels])
        return data   

def load(data_zip: str, data_set: str, lang: str, subtask: int = 1):
    """
    Subtask is either 1 or 2.
    """
    path = f"{data_zip}_{data_set}_{lang}_{subtask}.pickle"
    if os.path.isfile(path):
        print("loading data from pickle "+ path)
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        print("loading from zip")
        data  = load_from_zip(data_zip, data_set, lang, subtask)
        print("write cache file to:" + path)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return data

def transform_to_language_model_file(data_zip: str,data_set:str, lang: str,result_path):
    data = load(data_zip,data_set,lang,subtask=1)    
    text_file = open(result_path, "w", encoding="utf-8")

    for document in tqdm(data):
        word_count = 0 # count words per line
        for word,sentence_end in zip(document[0],document[1]):
            if sentence_end == '1':
                word_count = 0
                text_file.write(word + "\n")
            else:
                word_count += 1
                text_file.write(word + " ")

    text_file.close()

def data_augmentation(data,data_proportion = 1.0 ):
    """
        Perform data augmentation for task 1. 
        Recombines random sentences to new documents. 
    """
    print("running data augmentaion")
    sentences = []
    for document in tqdm(data):
        words, labels = [], []
        for word,sentence_end in zip(document[0],document[1]):
            words += [word]
            labels += [sentence_end]
            if sentence_end == '1':
                sentences+=[[words,labels]]
                words, labels = [], []
    random.shuffle(sentences)         
    if data_proportion <= 1.0:
        sentences = sentences[:int(len(sentences)*data_proportion)]
    else:
        sentences = sentences * int(data_proportion)
    result = []

    pbar = tqdm(total=len(sentences))
    while len(sentences) > 0:
        new_sentences = [[],[]]    
        random_len = random.randrange(3,30)
        for i in range(0,random_len,1):            
            sentence = sentences.pop(0)
            new_sentences[0] += sentence[0]
            new_sentences[1] += sentence[1]
            if len(sentences) == 0:
                break
        result += [new_sentences]    
        pbar.update(random_len)
    pbar.close()
    print("done.")
    return result


if __name__ =="__main__":
    #data = load("data/sepp_nlg_2021_train_dev_data_v5.zip","train","de",subtask=2)  
    data = load("data/leipzig_aug_de.zip","aug","de",subtask=1)  
    #data_aug = data_augmentaion(data[:50])
    #print(data_aug)

    classes = {}
    for item in data:
        for label in item[1]:
            if label in classes:
                classes[label] +=1 
            else:
                classes[label] = 1
    import pprint
    pprint.pprint(classes)
    #transform_to_language_model_file("data/sepp_nlg_2021_train_dev_data.zip","train","de", "data/sepp_nlg_train_de2.txt")
    #transform_to_language_model_file("data/sepp_nlg_2021_train_dev_data.zip","dev","de", "data/sepp_nlg_dev_de2.txt")