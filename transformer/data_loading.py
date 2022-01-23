from tqdm import tqdm
from datasets import Dataset, Features, ClassLabel, Value

def tokenize_and_align_data(tokenizer, data, stride=0):
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


def to_dataset(tokenizer, data, stride=0):
    labels, token_type_ids, input_ids, attention_masks = [],[],[],[]
    for item in tqdm(data):
        result = tokenize_and_align_data(tokenizer, item,stride=stride)        
        labels += result['labels']
        token_type_ids += result['token_type_ids']
        input_ids += result['input_ids']
        attention_masks += result['attention_mask']
    return Dataset.from_dict({'labels': labels, 'token_type_ids':token_type_ids, 'input_ids':input_ids, 'attention_mask':attention_masks})