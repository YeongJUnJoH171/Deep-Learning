
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split

import torchtext, torchdata
from torchtext.datasets import IMDB

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset


def text_transform(x, vocab, tokenizer, max_len):
    vlist = [vocab[token] for token in tokenizer(x)]
    # if the length of sequence exceeds max_len, return None
    if len(vlist) > max_len:
        return None
    vlist = vlist + [vocab['<PAD>']]*(max_len-len(vlist))
    return  vlist

def label_transform(x):
    return 1 if x == 'pos' else 0

def build_vocab(datasets, tokenizer):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)


class IMDB_tensor_dataset:
    def __init__(self, max_seq_len = 384, max_tokens = 20000):
        train_iter, test_iter = IMDB(split=('train','test'))

        self.tokenizer = get_tokenizer('basic_english')

        self.vocab = build_vocab_from_iterator(build_vocab([train_iter, test_iter], self.tokenizer), 
                                        min_freq=3, max_tokens =  max_tokens,
                                        specials=['<UNK>', '<PAD>'])
                                                
        self.vocab.set_default_index(self.vocab["<UNK>"])

        self.src_stoi = self.vocab.get_stoi()
        
        self.src_itos = self.vocab.get_itos()
        
        self.max_seq_len = max_seq_len
        
        train_set = to_map_style_dataset(train_iter)
        test_set = to_map_style_dataset(test_iter)

        datasets = [train_set, test_set]

        dsets = []
        for dataset in datasets:
            label_list, text_list = [], []
            for (_label, _text) in dataset:
                txt = text_transform(_text, self.vocab, self.tokenizer, max_seq_len)
                if txt is None:
                    continue
                processed_text = torch.tensor(txt)
                text_list.append(processed_text)
                label_list.append(label_transform(_label))
            X=pad_sequence(text_list, padding_value=self.src_stoi['<PAD>'], batch_first=True)
            y=torch.tensor(label_list)
            dsets.append(TensorDataset(X,y))

        self.train_dataset = dsets[0]
        self.test_dataset = dsets[1]
        
    def get_dataset(self):
        
        return self.train_dataset, self.test_dataset
