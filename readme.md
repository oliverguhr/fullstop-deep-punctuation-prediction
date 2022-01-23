# FullStop: Multilingual Deep Models for Punctuation Prediction

## Inference Sample

https://user-images.githubusercontent.com/3495355/150677531-13f2037d-8673-4e34-8769-0da1784c2fe7.mp4

This model predicts the punctuation of English, Italian, French and German texts.
 *Please note that the Europarl Dataset consists of political speeches. Therefore the model might perform differently on texts from other domains.*

The model restores the following punctuation markers: **"." "," "?" "-" ":"**

```python
from transformers import pipeline
pipe = pipeline("token-classification", "oliverguhr/fullstop-punctuation-multilang-large")
pipe(["My name is Clara and I live in Berkeley California"])
```

## Results 

The performance differs for the single punctuation markers as hyphens and colons, in many cases, are optional and can be substituted by either a comma or a full stop. The model achieves the following F1 scores for the different languages:

| Label         | EN    | DE    | FR    | IT    |
| ------------- | ----- | ----- | ----- | ----- |
| 0             | 0.991 | 0.997 | 0.992 | 0.989 |
| .             | 0.948 | 0.961 | 0.945 | 0.942 |
| ?             | 0.890 | 0.893 | 0.871 | 0.832 |
| ,             | 0.819 | 0.945 | 0.831 | 0.798 |
| :             | 0.575 | 0.652 | 0.620 | 0.588 |
| -             | 0.425 | 0.435 | 0.431 | 0.421 |
| macro average | 0.775 | 0.814 | 0.782 | 0.762 |

## Reproduce

In order to reproduce our experiments you can execute following code:

```
# setup
pip install -r requirements.txt
sh download-dataset.sh

# training
python model_test_suite.py -task 1 
python model_test_suite.py -task 2

# test
sh predict_all_task_1_models.sh
sh predict_all_task_2_models.sh
```

### Competition Website

https://sites.google.com/view/sentence-segmentation/

### Data

https://drive.switch.ch/index.php/s/g3fMhMZU2uo32mf

## Tested Models

English

* Distillbert
* bert-base-uncased
* bert-large-uncased 
* electra 
* roberta-base
* roberta-large 
* xlm-roberta-base
* xlm-roberta-large

German

* bert-base-multilingual-uncased
* dbmdz/bert-base-german-uncased
* distilbert-base-german-cased
* deepset/gbert-base
* uklfr/gottbert-base ->????

French

* bert-base-multilingual-uncased
* flaubert/flaubert_base_uncased
* camembert-base

Italian

* bert-base-multilingual-uncased
* dbmdz/bert-base-italian-cased
* dbmdz/bert-base-italian-uncased
* dbmdz/bert-base-italian-xxl-uncased
* dbmdz/electra-base-italian-xxl-cased-generator

en+de+fr+it

* bert-base-multilingual-uncased
* xlm-roberta-base
* xlm-roberta-large




## Cite us

```
@article{guhr-EtAl:2021:fullstop,
  title={FullStop: Multilingual Deep Models for Punctuation Prediction},
  author    = {Guhr, Oliver  and  Schumann, Anne-Kathrin  and  Bahrmann, Frank  and  Böhme, Hans Joachim},
  booktitle      = {Proceedings of the Swiss Text Analytics Conference 2021},
  month          = {June},
  year           = {2021},
  address        = {Winterthur, Switzerland},
  publisher      = {CEUR Workshop Proceedings},  
  url       = {http://ceur-ws.org/Vol-2957/sepp_paper4.pdf}
}
```



