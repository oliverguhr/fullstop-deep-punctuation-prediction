from transformers import pipeline
from dataset import load
import io
import os
from typing import List
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm


def predict_sent_end(model: str, data_zip: str, lang: str, data_set: str, outdir: str,task:str, overwrite: bool = True) -> None:
    outdir = os.path.join(outdir, lang, data_set)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f'using model {model}')
    pipe = pipeline("ner", model = model, grouped_entities=False, device=0)


    with ZipFile(data_zip, 'r') as zf:
        fnames = zf.namelist()
        relevant_dir = os.path.join('sepp_nlg_2021_data', lang, data_set)
        tsv_files = [
            fname for fname in fnames
            if fname.startswith(relevant_dir) and fname.endswith('.tsv')
        ]

        for i, tsv_file in enumerate(tqdm(tsv_files), 1):
            if not overwrite and Path(os.path.join(outdir, os.path.basename(tsv_file))).exists():
                continue

            with io.TextIOWrapper(zf.open(tsv_file), encoding="utf-8") as f:
                tsv_str = f.read()
            lines = tsv_str.strip().split('\n')
            rows = [line.split('\t') for line in lines]            
            words = [row[0] for row in rows]            
            
            lines = predict(pipe,words,task)
            with open(os.path.join(outdir, os.path.basename(tsv_file)), 'w',
                      encoding='utf8') as f:
                f.writelines(lines)        

def overlap_chunks(lst, n, stride=0):
    """Yield successive n-sized chunks from lst with stride length of overlap."""
    for i in range(0, len(lst), n-stride):
            yield lst[i:i + n]

label_2_id = {"0":0, ".":1, ",":2, "?":3, "-":4, ":":5}
id_2_label = list(label_2_id.keys())

def map_label_task_2(label):
    label_id = int(label[-1])
    return id_2_label[label_id]

def map_label_task_1(label):
    label_id = int(label[-1])
    # this way we can use task 2 models for task 1.
    # we set just set anything other than . to class 0
    if label_id != 1:
        label_id = 0
    return label_id

def predict(pipe,words, task):
    overlap = 5
    chunk_size = 230
    if len(words) <= chunk_size:
        overlap = 0

    batches = list(overlap_chunks(words,chunk_size,overlap))

    # if the last batch is smaller than the overlap, 
    # we can just remove it
    if len(batches[-1]) <= overlap:
        batches.pop()

    tagged_words = []     
    for batch in tqdm(batches):
        # use last batch completly
        if batch == batches[-1]: 
            overlap = 0
        text = " ".join(batch)
        #text = text.replace(" \xad","").replace("\xad","")
        result = pipe(text)      
        assert len(text) == result[-1]["end"], "chunk size too large, text got clipped"
            
        char_index = 0
        result_index = 0
        for word in batch[:len(batch)-overlap]:                
            char_index += len(word) + 1
            # if any subtoken of an word is labled as sentence end
            # we label the whole word as sentence end        
            label = 0
            while result_index < len(result) and char_index > result[result_index]["end"] :
                #label += 0 if result[result_index]['entity']  == 'LABEL_0' else 1
                if task == "1":
                    label = map_label_task_1(result[result_index]['entity'])
                if task == "2":
                    label = map_label_task_2(result[result_index]['entity'])
                result_index += 1
            #if label > 1: # todo: we should not need this line. please check
            #    print("i should be not needed")
            #    label = 1
            if task == "1":
                tagged_words += [f"{word}\t{label}\n"]
            if task == "2":
                tagged_words += [f"{word}\t-\t{label}\n"]
    
    if len(tagged_words) == len(words):
        # tracing script to find predicton errors
        for i,x in enumerate(zip(tagged_words,words)):
            if x[0].startswith(x[1]) == False:                
                print(i,x)
    assert len(tagged_words) == len(words)
    return tagged_words

if __name__ == '__main__':    
    import argparse
    parser = argparse.ArgumentParser(description='spaCy baseline for subtask 1 of SEPP-NLG 2021')
    parser.add_argument("data_zip", help="path to data zip file, e.g. 'data/sepp_nlg_2021_train_dev_data.zip'")
    parser.add_argument("language", help="target language ('en', 'de', 'fr', 'it'; i.e. one of the subfolders in the zip file's main folder)")
    parser.add_argument("data_set", help="dataset to be evaluated (usually 'dev', 'test'), subfolder of 'lang'")
    parser.add_argument("outdir", help="folder to store predictions in, e.g. 'data/predictions' (language and dataset subfolders will be created automatically)")
    parser.add_argument("model",help="path to transformers model")  
    parser.add_argument("task",help="task one or two")
    args = parser.parse_args()    
    predict_sent_end(args.model,args.data_zip, args.language, args.data_set, args.outdir,args.task, True)
