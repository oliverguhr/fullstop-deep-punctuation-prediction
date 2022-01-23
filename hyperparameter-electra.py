from datasets import load_dataset, load_metric
from dataset import load
from datasets import Dataset, Features, ClassLabel, Value
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import numpy.ma as ma
from transformers import DataCollatorForTokenClassification
from tqdm import tqdm
import datetime

task = "task1" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "german-nlp-group/electra-base-german-uncased"
#model_checkpoint = "dbmdz/bert-base-german-uncased"#"distilbert-base-german-cased"
run_name= f"{model_checkpoint}-{task}"
run_name = run_name.replace("/","-") + " " + str(datetime.datetime.now())[:-7]
batch_size = 8
label_all_tokens = True
data_factor = 1 # train and test on x percent of the data

## load data

val_data = load("data/sepp_nlg_2021_train_dev_data.zip","dev","de")
train_data = load("data/sepp_nlg_2021_train_dev_data.zip","train","de")

## tokenize data
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,strip_accent=False)

def tokenize_and_align_data(data,stride=0):
    tokenizer_settings = {'is_split_into_words':True,'return_offsets_mapping':True, 
                            'padding':False, 'truncation':True, 'stride':stride, 
                            'max_length':tokenizer.model_max_length, 'return_overflowing_tokens':True}
    tokenized_inputs = tokenizer(data[0], **tokenizer_settings)

    labels = []
    for i,document in enumerate(tokenized_inputs.encodings):
       doc_encoded_labels = []
       last_word_id = None
       for word_id  in document.word_ids:            
           if word_id == None: #or last_word_id == word_id:
               doc_encoded_labels.append(-100)        
           else:
               #document_id = tokenized_inputs.overflow_to_sample_mapping[i]
               #label = examples[task][document_id][word_id]
               label = data[1][word_id]
               doc_encoded_labels.append(int(label))
           last_word_id = word_id
       labels.append(doc_encoded_labels)
    
    tokenized_inputs["labels"] = labels    
    return tokenized_inputs


def to_dataset(data,stride=0):
    labels, token_type_ids, input_ids, attention_masks = [],[],[],[]
    for item in tqdm(data):
        result = tokenize_and_align_data(item,stride=stride)        
        labels += result['labels']
        token_type_ids += result['token_type_ids']
        input_ids += result['input_ids']
        attention_masks += result['attention_mask']
    return Dataset.from_dict({'labels': labels, 'token_type_ids':token_type_ids, 'input_ids':input_ids, 'attention_mask':attention_masks})

train_data = train_data[:int(len(train_data)*data_factor)] # limit data to x%

print("tokenize training data")
tokenized_dataset_train = to_dataset(train_data,stride=100)
del train_data

print("tokenize validation data")
val_data = val_data[:int(len(val_data)*(data_factor*5))] # limit data to x%
tokenized_dataset_val = to_dataset(val_data)
del val_data

## metrics 

def compute_metrics_sklearn(pred):    
    mask = np.less(pred.label_ids,0)    # mask out -100 values
    labels = ma.masked_array(pred.label_ids,mask).compressed() 
    preds = ma.masked_array(pred.predictions.argmax(-1),mask).compressed() 

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')  
    acc = accuracy_score(labels, preds)    
    return {     
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy':acc,        
    }

## train model



args = TrainingArguments(
    output_dir=f"models/{run_name}/checkpoints",
    run_name=run_name,    
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,    
    adafactor=False, 
    #weight_decay=0.005,
    warmup_steps=50,
    #lr_scheduler_type="cosine",
    #report_to=["tensorboard"],
    #logging_dir='runs/'+run_name,            # directory for storing logs
    #logging_first_step=True,
    #logging_steps=500,
    save_steps= 10000,
    save_total_limit=10,
    seed=100, 
    fp16=True   
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def model_init():
    return AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2)
    #return AutoModelForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)

trainer = Trainer(
    model_init=model_init,
    args = args,    
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_sklearn
)

trainer.train()

#trainer.save_model(f"models/{run_name}/final")
import gc
import torch

def my_hp_space(trial):    
    gc.collect()
    torch.cuda.empty_cache()
    return {        
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1,5),
#        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 10]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1,2,4,8]),
    }
#
#trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space)

#def my_hp_space_ray(trial):
#    from ray import tune
#
#    return {
#        "learning_rate": tune.loguniform(1e-4, 1e-2),
#        "num_train_epochs": tune.choice(range(1, 6)),
#        "seed": tune.choice(range(1, 41)),
#        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
#    }
#
#trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space)

def my_objective(metrics):
    # Your elaborate computation here
    return metrics['eval_f1']

result = trainer.hyperparameter_search(direction="maximize",n_trials=100,hp_space=my_hp_space, compute_objective=my_objective)

print("---- result  ----")
print(result)

with open("hyperparamert-electra.txt", "w") as f:
    print(result,file=f)
    f.close()
#
#BestRun(run_id='1', objective=0.9412869643435874, hyperparameters={'learning_rate': 0.0001945679450337696, 'num_train_epochs': 4, 'per_device_train_batch_size': 8, 'weight_decay': 1.4664229179484889e-09, 'adam_epsilon': 2.178754400098753e-07, 'gradient_accumulation_steps': 4})
