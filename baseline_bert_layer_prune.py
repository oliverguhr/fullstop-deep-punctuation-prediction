from datasets import load_dataset, load_metric
from datasets import Dataset, Features, ClassLabel, Value
from dataset import load, data_augmentation
from transformers import AutoTokenizer, Adafactor, AutoConfig, BertConfig
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import numpy.ma as ma
from transformers import DataCollatorForTokenClassification
from tqdm import tqdm
import datetime

task = 1 
number_of_layers = 12
#model_checkpoint = "german-nlp-group/electra-base-german-uncased"
model_checkpoint = "bert-base-uncased"#"distilbert-base-german-cased"
run_name= f"{model_checkpoint}-{task}-{number_of_layers}layer-10-data"
run_name = run_name.replace("/","-") + " " + str(datetime.datetime.now())[:-7]
batch_size = 8
label_all_tokens = True
data_factor = 0.1 # train and test on x percent of the data
sequenz_length = 512

## load data

val_data = load("data/sepp_nlg_2021_train_dev_data_v5.zip","dev","en",subtask=task)
train_data = load("data/sepp_nlg_2021_train_dev_data_v5.zip","train","en",subtask=task)

## tokenize data
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_data(data,stride=0):
    tokenizer_settings = {'is_split_into_words':True,'return_offsets_mapping':True, 
                            'padding':False, 'truncation':True, 'stride':stride, 
                            'max_length':sequenz_length, 'return_overflowing_tokens':True}
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
#train_data += data_augmentation(train_data,.5)
print("tokenize training data")
tokenized_dataset_train = to_dataset(train_data,stride=400)
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

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')  
    acc = accuracy_score(labels, preds)    
    return {     
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy':acc,        
    }

## train model
config = AutoConfig.from_pretrained(model_checkpoint)
config.num_hidden_layers = number_of_layers
config.num_labels=2
#config.max_position_embeddings = sequenz_length
print("-----------------")
print(config)
#config.num_attention_heads = 6
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,config = config)
print("Model Parameters " + str(model.num_parameters()))

# freez layers
#for name, param in model.named_parameters():
#	if 'classifier' not in name:
#		param.requires_grad = False

#model.prune_heads({0:[0,1,2,3,4,5,6],1:[0,1,2,3,4,5,6],2:[0,1,2,3,4,5,6],3:[0,1,2,3,4,5,6],4:[0,1,2,3,4,5,6],5:[0,1,2,3,4,5,6]})
args = TrainingArguments(
    output_dir=f"models/{run_name}/checkpoints",
    run_name=run_name,    
    evaluation_strategy = "epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    adafactor=True, 
    learning_rate=4e-5,    
    warmup_steps=50,    
    #weight_decay=0.035,
    #adam_epsilon= 2e-09,
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
    compute_metrics=compute_metrics_sklearn,
)
trainer.train()

trainer.save_model(f"models/{run_name}/final")