# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Inicializando tokenizer e stop words

# %%
import os
import re
# import spacy
import string
import numpy as np
import pandas as pd
from typing import Union, List
from collections import Counter
# from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ### Criando modelo e carregando stop-words

# %%
stop_words = np.array(['because', '’m', 'name', 'therein', '’ll', 'already', 'that',
       'hundred', 'her', 'cannot', 'before', 'ever', 'regarding', 'get',
       'these', 'as', 'if', 'when', 'onto', 'ours', 'everything', '‘re',
       'from', 'whereby', 'side', 'and', 'do', 'must', 'three',
       'throughout', 'rather', 'its', 'was', 'amount', 'whose', 'how',
       'hereby', 'top', 'see', 'quite', 'thus', 'further', 'last',
       'myself', 'enough', 'himself', 'formerly', 'herself', 'more',
       'whereafter', 'per', 'yourselves', 'us', 'various', 'everywhere',
       'five', 'next', 'below', 'she', 'through', 'once', 'eight',
       'which', 'most', 'be', 'above', 'whither', 'wherein', 'up',
       'fifty', 'back', 'in', 'seeming', '’ve', 'after', 'full', 'mine',
       'yours', 'here', 'out', 'those', 'n‘t', 'eleven', 'all', 'same',
       'is', 'however', 'became', 'not', 'either', 'within', 'a', 'part',
       'nobody', 'did', 'without', 'many', 'but', 'might', 'nine', 'nor',
       'twenty', 'whatever', '’s', 'go', 'former', 'no', 'so', "'ll",
       'beside', 'therefore', 'about', 'hers', '‘s', 'third', 'much',
       "n't", 'everyone', 'own', 'over', '‘ve', "'ve", 'any', 'other',
       'during', 'else', 'still', 'towards', 'bottom', 'his', 'together',
       'perhaps', 'though', 'whole', 'besides', 'yourself', 'who',
       'using', 'noone', 'made', 'been', 'alone', 'whom', 'around',
       'please', 'along', 'are', 'thereupon', 'such', 'latterly', 'very',
       'sixty', 'anywhere', 'an', 'am', 'mostly', 'since', 'were',
       'become', 'first', 'less', 'moreover', '‘d', 'even', 'does', '’d',
       'each', 'now', 'while', 'indeed', 'our', 'becoming', 'empty',
       'some', 'unless', 'their', 'both', 'give', 'your', 'anything',
       'whereupon', 'nothing', 'of', 'neither', 'upon', 'beyond', 'least',
       'say', 'would', '‘ll', 'just', 'every', 'hereupon', 'via', 'down',
       'me', 'although', 'into', 'almost', 'seems', 'my', 'becomes',
       'whereas', 'latter', 'seem', 'then', 'he', 'serious', 'for',
       'front', 'the', 'can', 'few', 're', 'you', 'by', 'could', '’re',
       'to', 'six', 'elsewhere', 'than', 'well', "'d", 'namely', 'under',
       'i', 'someone', 'until', 'anyhow', 'move', 'itself', 'whether',
       'put', 'hence', 'toward', 'never', 'often', 'thru', 'or', 'with',
       'meanwhile', 'on', 'off', 'at', 'twelve', 'seemed', 'four', 'used',
       'done', 'two', 'otherwise', 'beforehand', 'hereafter', 'amongst',
       'across', 'between', 'due', 'they', 'call', 'may', 'afterwards',
       '‘m', "'s", 'one', 'wherever', 'we', 'always', 'has', 'against',
       'doing', 'being', 'n’t', 'another', 'should', 'ca', 'except',
       'thereby', 'what', 'him', 'forty', 'keep', 'show', 'themselves',
       'sometimes', 'whence', 'anyone', 'fifteen', 'it', 'somewhere',
       'also', 'take', 'nowhere', 'this', 'nevertheless', 'anyway',
       'ourselves', 'will', 'something', 'have', 'there', 'thence', 'why',
       "'re", 'ten', 'too', 'thereafter', 'none', 'make', 'somehow',
       'only', 'others', "'m", 'whoever', 'several', 'sometime', 'among',
       'had', 'behind', 'whenever', 'yet', 'them', 'really', 'again',
       'where', 'herein'], dtype='<U12')

# %% [markdown]
# ### Carregando o dataset

# %%
imdb_dataset_filepath = os.path.abspath("imdb.csv")
imdb_dataset = pd.read_csv(imdb_dataset_filepath, names=["Text", "Prediction"], sep="\t")
imdb_dataset.head()

# %% [markdown]
# ### Criação do BoW

# %%
class SIA:
    bow: np.ndarray
    dbow: np.ndarray
    dbow_0: np.ndarray
    dbow_1: np.ndarray
    word_count: Counter
    sentences: List[str]
    processed_sentences: List[str]
    classification: List[int]
    detectors: List[np.ndarray]
    vectorized_format_text: object

    def __init__(self, sentences: Union[List[str], np.ndarray], classification: Union[List[int], np.ndarray]) -> None:
        self.sentences = sentences
        self.classification = classification
        self.vectorized_format_text = np.vectorize(self.format_text)

    def format_text(self, text: str) -> str:
        # remove caracteres que não sejam letras e numeros
        fixed_text = re.sub(r"[^A-Za-z\s]", "", text)
        fixed_text = re.sub(r"\s{2,}", r" ", fixed_text).casefold()
        fixed_text = re.sub(r"^\s+|\s+$", "", fixed_text)
        tokens = fixed_text.split()
        words = [token for token in tokens if token not in stop_words]
        return " ".join(words)
    
    def pre_process(self) -> None:
        self.processed_sentences = self.vectorized_format_text(self.sentences)
        self.set_bow()
        self.set_dbow()

    def set_bow(self) -> None:
        all_tokens = np.array(("".join(self.processed_sentences).split()))
        self.word_count = Counter(all_tokens)
        self.bow = np.array([*self.word_count.keys()])
        print(f"bow with {self.bow.shape[0]} words")
    
    def set_dbow(self) -> None:
        self.dbow = self.generate_dbow(self.sentences.shape[0], self.processed_sentences)
        self.dbow_0 = self.dbow[np.where(self.classification == 0)]
        self.dbow_1 = self.dbow[np.where(self.classification == 1)]
    
    def generate_dbow(self, X_size: int, sentences: Union[List[str], np.ndarray]) -> np.ndarray:
        dbow = np.zeros((X_size, self.bow.shape[0]))
        for i in range(X_size):
            sentence_counter = Counter(sentences[i].split())
            for j in range(dbow.shape[1]):
                dbow[i][j] = 1 if sentence_counter[self.bow[j]] > 0 else 0
        return dbow

    def generate_detectors(self, number_of_detectors: int) -> None:
        self.detectors = []
        while (len(self.detectors) < number_of_detectors):
            candidate_detector = self.generate_candidate_detector()
            for i in range(self.dbow_0.shape[0]):
                if(self.match(self.dbow_0[i], candidate_detector)):
                    break
            else:
                self.detectors.append(candidate_detector)

    def generate_candidate_detector(self, word_probability = 0.18) -> np.ndarray:
        detector = np.zeros(self.bow.shape[0])
        for i in range(detector.shape[0]):
            if (np.random.rand() < word_probability):
                detector[i] = 1
        return detector

    def match(self, match_set: np.ndarray, detector: np.ndarray, threshold = 5) -> bool:
        return ((match_set == detector).astype(int).sum() >= threshold)

    def detect(self, sentences: np.ndarray) -> np.ndarray:
        classification_results = np.zeros(sentences.shape[0], np.int8)
        pre_processed_sentences = self.vectorized_format_text(sentences)
        detect_dbow = self.generate_dbow(pre_processed_sentences.shape[0], pre_processed_sentences)
        for i in range(detect_dbow.shape[0]):
            for detector in self.detectors:
                print(f"dbow {detect_dbow[i]} detector {detector}")
                if(self.match(detect_dbow[i], detector)):
                    classification_results[i] = 1
                    break
        return classification_results

    def export(self, file_path = os.getcwd()) -> None:
        with open(os.path.abspath(os.path.join(file_path, "bow.txt")), "w+") as f:
            f.write(str(list(self.bow)))
        with open(os.path.abspath(os.path.join(file_path, "dbow.txt")), "w+") as f:
            f.write(str(list(self.dbow)))


# %%
s = SIA(imdb_dataset["Text"].to_numpy(), imdb_dataset["Prediction"].to_numpy())


# %%
s.pre_process()


# %%
print(s.sentences[0])
print(s.processed_sentences[0])


# %%
s.word_count.most_common(10)


# %%
s.generate_detectors(10)


# %%
s.detectors


# %%
s.detect(imdb_dataset.Text)


# %%
pre_processed_sentences = s.vectorized_format_text(imdb_dataset["Text"])
new_dbow = s.generate_dbow(pre_processed_sentences.shape[0], pre_processed_sentences)


# %%
pre_processed_sentences[0]


# %%
s.match(new_dbow[0], )


# %%
s.bow[np.where(new_dbow[0]>0)]


# %%



# %%


# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=993f98d4-e474-42c5-8e20-241471545034' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>

