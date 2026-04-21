from datasets import load_dataset, load_metric, concatenate_datasets
from dataset import load
from datasets import Dataset, Features, ClassLabel, Value, concatenate_datasets
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import numpy.ma as ma
from transformers import DataCollatorForTokenClassification
from tqdm import tqdm
import datetime
import random
from tools import print_cm
from multiprocessing import Pool
from functools import partial 
import psutil
import os
import zipfile
import gc
from typing import List

class ModelTrainer():
    def __init__(self, task:int, model:str,run_name:str, data: List[str], data_percentage:float, tokenize_min_batch_size:int, stride:int, use_token_type_ids:bool, opimizer_config, tokenizer_config, languages, batches_ending_on_question_mark_percent:float = 0.0, do_hyperparameter_search = False, resume = False, **args):
        self.task = task 
        self.model_checkpoint = model
        self.run_name = run_name
        self.batch_size = 8
        self.label_all_tokens = True
        self.data_archive_paths = data
        self.data_factor = data_percentage # train and test on x percent of the data
        self.tokenize_min_batch_size = tokenize_min_batch_size
        if batches_ending_on_question_mark_percent < 0 or batches_ending_on_question_mark_percent > 100:
            raise ValueError("batches_ending_on_question_mark_percent must be between 0 and 100")
        self.batches_ending_on_question_mark_percent = batches_ending_on_question_mark_percent
        self.stride = stride
        self.opimizer_config = opimizer_config
        self.tokenizer_config = tokenizer_config
        self.languages = languages
        self.use_token_type_ids = use_token_type_ids
        self.do_hyperparameter_search = do_hyperparameter_search
        self.resume = resume
        
        if self.task == 1:    
            self.label_2_id = {"0":0, "1":1}
        else:
            self.label_2_id = {"0":0, ".":1, ",":2, "?":3, "-":4, ":":5} 
            
        self.id_2_label = list(self.label_2_id.keys())        

    def should_target_question_mark_batch(self, tokenized_batch_count, question_mark_batch_count):
        if self.batches_ending_on_question_mark_percent <= 0:
            return False

        next_batch_count = tokenized_batch_count + 1
        question_mark_fraction = self.batches_ending_on_question_mark_percent / 100.0
        return (question_mark_batch_count + 1) / next_batch_count <= question_mark_fraction

    def find_last_question_mark_index(self, labels):
        min_split_index = min(int(0.75 * self.tokenize_min_batch_size), len(labels))
        for i in range(len(labels) - 1, min_split_index - 1, -1):
            if labels[i] == "?":
                return i
        return None

    def trim_batch(self, batch, tokenized_batch_count, question_mark_batch_count):
        if self.should_target_question_mark_batch(tokenized_batch_count, question_mark_batch_count):
            question_mark_index = self.find_last_question_mark_index(batch[1])
            if question_mark_index is not None:
                split_at = question_mark_index + 1
                return [batch[0][:split_at], batch[1][:split_at]], [batch[0][split_at:], batch[1][split_at:]]

        return batch, [[], []]
   
    def tokenize_and_align_data(self,data,max_length=512,stride=0):
        if self.model_checkpoint == "camembert/camembert-large":
            # this model has a wrong maxlength value, so we need to set it manually
            self.tokenizer.model_max_length = 512 
            
        tokenizer_settings = {'is_split_into_words':True,'return_offsets_mapping':True, 
                                'padding':False, 'truncation':True, 'stride':stride, 
                                'max_length':max_length, 'return_overflowing_tokens':True}

        tokenized_inputs = self.tokenizer(data[0], **tokenizer_settings)
        
        labels = []
        for i,document in enumerate(tokenized_inputs.encodings):
            doc_encoded_labels = []
            last_word_id = None
            for word_id in document.word_ids:            
                if word_id == None: #or last_word_id == word_id:
                    doc_encoded_labels.append(-100)        
                else:
                    label = data[1][word_id]
                    doc_encoded_labels.append(self.label_2_id[label])
                last_word_id = word_id
            labels.append(doc_encoded_labels)
        
        tokenized_inputs["labels"] = labels    
        return tokenized_inputs


    def to_dataset(self, data, max_length=512, stride=0):
        labels, token_type_ids, input_ids, attention_masks = [],[],[],[]
        print(f"to_dataset: data_len: {len(data)}, max_length: {max_length}, stride: {stride}, tokenize_min_batch_size:{self.tokenize_min_batch_size}, batches_ending_on_question_mark_percent:{self.batches_ending_on_question_mark_percent}")

        total_elements = sum(len(item[0]) for item in data)
        count_elements_to_clear = 0
        tokenized_batch_count = 0
        question_mark_batch_count = 0
         
        large = [[], []]
        for i,item in enumerate(tqdm(data)):
            count_elements_to_clear += len(item[0])

            large[0].extend(item[0])
            large[1].extend(item[1])

            # Aggregate the data items before tokenizing in order to avoid overfitting
            # to the end labels of individual TSV files.
            is_last_item = i == len(data) - 1
            while len(large[0]) > self.tokenize_min_batch_size or (is_last_item and len(large[0]) > 0):
                if len(large[0]) > self.tokenize_min_batch_size:
                    batch, large = self.trim_batch(
                        large,
                        tokenized_batch_count,
                        question_mark_batch_count
                    )
                else:
                    batch, large = large, [[], []]

                result = self.tokenize_and_align_data(batch, max_length=max_length, stride=stride)

                labels.extend(result['labels'])
                if self.use_token_type_ids:
                    token_type_ids.extend(result['token_type_ids'])
                input_ids.extend(result['input_ids'])
                attention_masks.extend(result['attention_mask'])

                tokenized_batch_count += 1
                if batch[1] and batch[1][-1] == "?":
                    question_mark_batch_count += 1

                # Every so often free memory by setting to None the already processed elements of 'data' and forcing garbage collection
                if count_elements_to_clear > total_elements / 10:
                    data[:i] = [None] * i 
                    gc.collect()
                    count_elements_to_clear = 0
       
        gc.collect()

        print(f"use_token_type_ids: {self.use_token_type_ids}")
        print(f"tokenized_batch_count: {tokenized_batch_count}, question_mark_batch_count: {question_mark_batch_count}")
        if self.use_token_type_ids:
            return Dataset.from_dict({'labels': labels, 'token_type_ids':token_type_ids, 'input_ids':input_ids, 'attention_mask':attention_masks})
        else:
            return Dataset.from_dict({'labels': labels, 'input_ids':input_ids, 'attention_mask':attention_masks})

    def compute_metrics_generator(self):    
        def metrics(pred):
            mask = np.less(pred.label_ids,0)    # mask out -100 values
            labels = ma.masked_array(pred.label_ids,mask).compressed() 
            preds = ma.masked_array(pred.predictions.argmax(-1),mask).compressed() 
            if self.task == 1:
                precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")  
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")  
                print("\n----- report -----\n")
                report = classification_report(labels, preds,target_names=self.label_2_id.keys())
                print(report)
                print("\n----- confusion matrix -----\n")
                cm = confusion_matrix(labels,preds,normalize="true")
                print_cm(cm,self.id_2_label)

            acc = accuracy_score(labels, preds)    
            return {     
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy':acc,        
            }
        return metrics   

    def load_language(self, language, data_type):
        data = []
        for archive_path in self.data_archive_paths:
            with zipfile.ZipFile(archive_path, 'r') as archive:
                has_data_for_language = any(f'/{language}/' in item for item in archive.namelist())

            if has_data_for_language:
                print (f"Loading {data_type} data for language '{language}' from {archive_path}")
                data += load(f"{archive_path}", data_type, language, subtask=self.task)
        return data
    
    def load_and_tokenize_language(self, language, data_type, max_length=512, stride=100):
        print (f"Loading and tokenizing {data_type} data for '{language}'") 
        data = self.load_language(language, data_type)
        print(f"Tokenizing {data_type} data for '{language}'")
        tokenized = self.to_dataset(data, max_length=max_length, stride=stride)
        
        # Free memory
        del data
        gc.collect()

        return tokenized

    def load_and_tokenize_parallel(self, data_type, num_procs=1,  max_length=512, stride=100):
        load_and_tokenize_language_partial = partial(self.load_and_tokenize_language, data_type=data_type, max_length=max_length, stride=stride)

        with Pool(num_procs) as pool:
            result = list(pool.map(load_and_tokenize_language_partial, self.languages))

        return concatenate_datasets(result)

    def run_training(self):
        val_data = []
        train_data = []
        print (self.languages)
        print("TASK " + str(self.task))
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint,**self.tokenizer_config)
       
        num_procs = max(psutil.cpu_count(logical=False) - 1, 1)
        print (f"Number of processes to use for loading and tokenizing: {num_procs}")
        print (f"Sride: {self.stride}")
        # The data is loaded from the archives listed in the 'data' array specified in the config in config.  
            
        print(f"data factor: {self.data_factor}")

        #tokenized_dataset_train = self.to_dataset(train_data, max_length=512, stride=100)
        tokenized_dataset_train = self.load_and_tokenize_parallel("train", num_procs=num_procs, max_length=512, stride=self.stride)
        
        print("Training data tokenized.")
        
        # Free memory
        del train_data
        gc.collect()

        if self.data_factor < 1.0:
            train_split = tokenized_dataset_train.train_test_split(train_size=self.data_factor)
            tokenized_dataset_train = train_split["train"]

        tokenized_dataset_train.shuffle(seed=42)

        # Load validation data
        tokenized_dataset_val = self.load_and_tokenize_parallel("dev", num_procs=num_procs, max_length=512, stride=self.stride)
        print("Validation data tokenized.")
        del val_data
        gc.collect() 
        
        ## train model
        args = TrainingArguments(
            output_dir=f"models/{self.run_name}/checkpoints",
            run_name=self.run_name,    
            evaluation_strategy = "epoch",
            learning_rate=4e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=self.opimizer_config["num_train_epochs"],
            adafactor=self.opimizer_config["adafactor"], 
            #weight_decay=0.005,
            #weight_decay=2.4793153505992856e-11,
            #adam_epsilon=5.005649261324263e-10,
            warmup_steps=50,
            #lr_scheduler_type="cosine",
            report_to=["tensorboard"],
            logging_dir='runs/'+self.run_name,            # directory for storing logs
            logging_first_step=True,
            logging_steps=100,
            save_steps=40000,
            save_total_limit=5,
            seed=16, 
            fp16=True   
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        def model_init():
            return AutoModelForTokenClassification.from_pretrained(self.model_checkpoint, num_labels=len(self.label_2_id))

        trainer = Trainer(
            model_init=model_init,
            args = args,    
            train_dataset=tokenized_dataset_train,
            eval_dataset=tokenized_dataset_val,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_generator()
        )

        if self.do_hyperparameter_search:
            print("----------hyper param search------------")
            return self.run_hyperparameter_search(trainer)
        else:
            trainer.train(resume_from_checkpoint=False)
            trainer.save_model(f"models/{self.run_name}/final")
            return trainer.state.log_history

    def run_hyperparameter_search(self, trainer):
            import gc
            import torch
            def my_hp_space(trial):    
                gc.collect()
                torch.cuda.empty_cache()
                return {        
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                    "num_train_epochs": trial.suggest_int("num_train_epochs", 1,5),
                    "seed": trial.suggest_int("seed", 1, 40),
                    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8]),
                    "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
                    "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
                    "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1,2,4,8]),
                }
            def my_objective(metrics):            
                return metrics['eval_f1']

            result = trainer.hyperparameter_search(direction="maximize",n_trials=200,hp_space=my_hp_space, compute_objective=my_objective)
            print("---hyper---")
            print(result)
            print("---hyper---")
            return result

if __name__ =="__main__":
    trainer = ModelTrainer(task=2,model="dbmdz/bert-base-italian-xxl-uncased",run_name="optim",data_percentage=0.1,use_token_type_ids=True, opimizer_config={"adafactor": False,"num_train_epochs": 3}, tokenizer_config={"strip_accent": True, "add_prefix_space":False}, languages=["it"], do_hyperparameter_search=True)
    result = trainer.run_training()

    print(result)
