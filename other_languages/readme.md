# How to train for other languages

## Create a dataset for your language

If Europarl dataset is available into your language, you can use the [SEPP-NLG-2021](https://github.com/dtuggener/SEPP-NLG-2021) scripts to create a tsv file from Europarl datasets.

If there is no Europarl dataset for your language, we created a small [tool](./create_shared_task_data_from_text_file.py) that takes as an input file a text file and produces a tsv format (SEPP NLG task) file suitable for training. This opens the posibility to use any plan text corpus for training.

In both cases, the dataset has to be placed in the directory _data_ with the name _sepp_nlg_2021_train_dev_data_v5.zip_. Inside the zip file, you will find the diretory structure _/sepp_nlg_2021_data/LANGUAGE_CODE/_ and then a _dev_ and _train_ directory.

## Defining your language and model

In the file _model_final_suite_task2.json_ you need to define your language and model. For Catalan language, this how it looks like:

```json
{   
    "tests":[   
        {
            "id":22,
            "task":2,
            "model": "softcatala/julibert",
            "languages":["ca"],
            "augmentation":[""],
            "data_percentage":1,
            "use_token_type_ids":false,
            "tokenizer_config":{"strip_accent":false, "add_prefix_space":true },
            "opimizer_config":{"adafactor":true, "num_train_epochs":2}
        }
    ]
}
```
The most important fields are _languages_, where you need to define your language and  _model_ where you need to define the base model.

## Training

Before starting the training make sure that the _model_final_suite_task2.json_ file is empty, looking like:

```json

{
    "tests": {}
}

```

You can start the training be running:

```python

python model_test_suite.py 2

```

