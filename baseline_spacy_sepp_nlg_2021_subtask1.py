"""
Baseline for subtask 1 of SEPP-NLG 2021: https://sites.google.com/view/sentence-segmentation/
Use spaCy to predict sentence boundaries
"""

__author__ = 'don.tuggener@zhaw.ch'

import io
import os
import spacy
from typing import List
from spacy.tokens import Doc
from pathlib import Path
from zipfile import ZipFile


class TokenizerFromList:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words: List[str]):
        spaces = [True for _ in range(len(words))]
        return Doc(self.vocab, words=words, spaces=spaces)


def predict_sent_end(data_zip: str, lang: str, data_set: str, outdir: str, overwrite: bool = False,
                     model_size: str = 'sm') -> None:
    outdir = os.path.join(outdir, lang, data_set)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if model_size == 'sm':
        if lang == 'en':
            spacy_model_name = 'en_core_web_sm'
        elif lang in {'de', 'fr', 'it'}:
            spacy_model_name = f'{lang}_core_news_sm'
        else:
            raise NotImplementedError
    elif model_size == 'lg':
        if lang == 'en':
            spacy_model_name = 'en_core_web_trf'
        elif lang == 'fr':
            spacy_model_name = 'fr_dep_news_trf'
        elif lang == 'it':
            spacy_model_name = 'it_core_news_lg'
        elif lang == 'de':
            spacy_model_name = 'de_dep_news_trf'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    print(f'using spaCy model {spacy_model_name}')
    nlp_pretok = spacy.load(spacy_model_name)
    nlp_pretok.tokenizer = TokenizerFromList(nlp_pretok.vocab)
    nlp_pretok.max_length = 1500000

    with ZipFile(data_zip, 'r') as zf:
        fnames = zf.namelist()
        relevant_dir = os.path.join('sepp_nlg_2021_data', lang, data_set)
        tsv_files = [
            fname for fname in fnames
            if fname.startswith(relevant_dir) and fname.endswith('.tsv')
        ]

        for i, tsv_file in enumerate(tsv_files, 1):
            print(i, tsv_file)

            if not overwrite and Path(os.path.join(outdir, os.path.basename(tsv_file))).exists():
                continue

            with io.TextIOWrapper(zf.open(tsv_file), encoding="utf-8") as f:
                tsv_str = f.read()
            lines = tsv_str.strip().split('\n')
            rows = [line.split('\t') for line in lines]
            tokens = [row[0] for row in rows]
            doc = nlp_pretok(tokens)

            with open(os.path.join(outdir, os.path.basename(tsv_file)), 'w',
                      encoding='utf8') as f:
                for sent in doc.sents:
                    sent_len = len(sent)
                    for i, tok in enumerate(sent):
                        if i == sent_len - 1:
                            f.write(f'{tok.text}\t1\n')
                        else:
                            f.write(f'{tok.text}\t0\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='spaCy baseline for subtask 1 of SEPP-NLG 2021')
    parser.add_argument("data_zip", help="path to data zip file, e.g. 'data/sepp_nlg_2021_train_dev_data.zip'")
    parser.add_argument("language", help="target language ('en', 'de', 'fr', 'it'; i.e. one of the subfolders in the zip file's main folder)")
    parser.add_argument("data_set", help="dataset to be evaluated (usually 'dev', 'test'), subfolder of 'lang'")
    parser.add_argument("outdir", help="folder to store predictions in, e.g. 'data/predictions' (language and dataset subfolders will be created automatically)")
    parser.add_argument("model_size", default='sm', help="spaCy model size (sm or lg)", nargs='?')
    args = parser.parse_args()
    predict_sent_end(args.data_zip, args.language, args.data_set, args.outdir, model_size=args.model_size)