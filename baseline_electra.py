from datasets import load_dataset, load_metric, concatenate_datasets
from dataset import load
from datasets import Dataset, Features, ClassLabel, Value
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

task = 1 # Should be one of "ner", "pos" or "chunk"
#dbmdz/electra-base-german-europeana-cased-discriminator
model_checkpoint ="german-nlp-group/electra-base-german-uncased" #"deepset/gelectra-large" #"german-nlp-group/electra-base-german-uncased"#"german-nlp-group/electra-base-german-uncased"
run_name= f"{model_checkpoint}-{task}-adamw-optimal-hyperparameter-v5data"
run_name = run_name.replace("/","-") + " " + str(datetime.datetime.now())[:-7]
batch_size = 4
label_all_tokens = True
data_factor = 0.1 # train and test on x percent of the data


if task == 1:    
    label_2_id = {"0":0, "1":1}
else:
    label_2_id = {"0":0, ".":1, ",":2, "?":3, "-":4, ":":5} 

id_2_label = list(label_2_id.keys())
## load data

val_data = load("data/sepp_nlg_2021_train_dev_data_v5.zip","dev","de",subtask=task)
train_data = load("data/sepp_nlg_2021_train_dev_data_v5.zip","train","de",subtask=task)
#aug_data = load("data/bundestag_aug.zip","aug","de",subtask=task)
#aug_data += load("data/leipzig_aug_de.zip","aug","de",subtask=task)
## tokenize data
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,strip_accent=True)

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
               doc_encoded_labels.append(label_2_id[label])
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

#train_data = train_data[:int(len(train_data)*data_factor)] # limit data to x%
#aug_data = aug_data[:int(len(aug_data)*data_factor)] # limit data to x%
print("tokenize training data")
tokenized_dataset_train = to_dataset(train_data,stride=100)
del train_data
#tokenized_dataset_aug = to_dataset(aug_data,stride=100)
#del aug_data
if data_factor < 1.0:
    train_split = tokenized_dataset_train.train_test_split(train_size=data_factor)
    tokenized_dataset_train = train_split["train"]
#    aug_split = tokenized_dataset_aug.train_test_split(train_size=data_factor)
#    tokenized_dataset_aug = aug_split["train"]

#tokenized_dataset_train = concatenate_datasets([tokenized_dataset_aug,tokenized_dataset_train])
tokenized_dataset_train.shuffle(seed=42)

print("tokenize validation data")
val_data = val_data[:int(len(val_data)*data_factor)] # limit data to x%
tokenized_dataset_val = to_dataset(val_data)
del val_data

## metrics 

def compute_metrics_sklearn(pred):    
    mask = np.less(pred.label_ids,0)    # mask out -100 values
    labels = ma.masked_array(pred.label_ids,mask).compressed() 
    preds = ma.masked_array(pred.predictions.argmax(-1),mask).compressed() 
    if task == 1:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")  
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")  
        print("\n----- report -----\n")
        report = classification_report(labels, preds,target_names=label_2_id.keys())
        print(report)
        print("\n----- confusion matrix -----\n")
        cm = confusion_matrix(labels,preds,normalize="true")
        print_cm(cm,id_2_label)

    acc = accuracy_score(labels, preds)    
    return {     
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy':acc,        
    }

## train model

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_2_id))

# best hyperparameters 200 step search
# hyperparameters={'learning_rate': 5.093122178478288e-05, 'num_train_epochs': 2, 'seed': 36, 'warmup_steps': 50, 'per_device_train_batch_size': 4, 'weight_decay': 2.1026179924358116e-11, 'adam_epsilon': 3.8026122222804776e-08})

args = TrainingArguments(
    output_dir=f"models/{run_name}/checkpoints",
    run_name=run_name,    
    evaluation_strategy = "epoch",
    learning_rate=5.093122178478288e-05,
    #learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    adafactor=True, 
    #weight_decay=0.005,
    weight_decay=2.1026179924358116e-11,
    adam_epsilon=3.8026122222804776e-08,
    warmup_steps=50,
    #lr_scheduler_type="cosine",
    report_to=["tensorboard"],
    logging_dir='runs/'+run_name,            # directory for storing logs
    logging_first_step=True,
    logging_steps=100,
    save_steps= 10000,
    save_total_limit=10,
    seed=16, 
    fp16=True   
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    args,    
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_sklearn
)

trainer.train()
trainer.save_model(f"models/{run_name}/final")