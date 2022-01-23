from datasets import load_dataset, load_metric
from datasets import Dataset, Features, ClassLabel, Value
from dataset import load, data_augmentation
from transformers import AutoTokenizer, Adafactor
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import numpy.ma as ma
from transformers import DataCollatorForTokenClassification
from tqdm import tqdm
import datetime

task = 2 
#model_checkpoint = "bert-base-german-cased" # "dbmdz/bert-base-german-uncased"
#"uklfr/gottbert-base"
model_checkpoint ="dbmdz/bert-base-german-uncased"#"models/sepp2021-de-512-full" # "dbmdz/bert-base-german-uncased"#"distilbert-base-german-cased"
run_name= f"{model_checkpoint}-{task}-sepp-adafactor-optim-hyperparameter-full"
run_name = run_name.replace("/","-") + " " + str(datetime.datetime.now())[:-7]
batch_size = 8
label_all_tokens = True
data_factor = 1 # train and test on x percent of the data

label_2_id = {"0":0, ".":1, ",":2, "?":3, "!":4, ";":5}

## load data

val_data = load("data/sepp_nlg_2021_train_dev_data.zip","dev","de",task)
train_data = load("data/sepp_nlg_2021_train_dev_data.zip","train","de",task)

## tokenize data
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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

train_data = train_data[:int(len(train_data)*data_factor)] # limit data to x%
#train_data += data_augmentation(train_data,.5)
print("tokenize training data")
tokenized_dataset_train = to_dataset(train_data,stride=100)
del train_data

print("tokenize validation data")
val_data = val_data[:int(len(val_data)*data_factor)] # limit data to x%
tokenized_dataset_val = to_dataset(val_data)
del val_data

## metrics 

def compute_metrics_sklearn(pred):    
    mask = np.less(pred.label_ids,0)    # mask out -100 values
    labels = ma.masked_array(pred.label_ids,mask).compressed() 
    preds = ma.masked_array(pred.predictions.argmax(-1),mask).compressed() 

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')  
    print("----- report -----")
    report = classification_report(labels, preds,target_names=label_2_id.keys())
    print(report)
 
    acc = accuracy_score(labels, preds)    
    return {     
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy':acc,        
    }

## train model

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_2_id))

args = TrainingArguments(
    output_dir=f"models/{run_name}/checkpoints",
    run_name=run_name,    
    evaluation_strategy = "epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    adafactor=False, 
    learning_rate=3.5e-5,    
    warmup_steps=50,    
    weight_decay=0.0088,
    adam_epsilon= 6e-08,
    #lr_scheduler_type="cosine",
    report_to=["tensorboard"],
    logging_dir='runs/'+run_name,            # directory for storing logs
    logging_first_step=True,
    logging_steps=100,
    save_steps= 10000,
    save_total_limit=10,
    seed=17, 
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
    compute_metrics=compute_metrics_sklearn,
)

trainer.train()
trainer.save_model(f"models/{run_name}/final")