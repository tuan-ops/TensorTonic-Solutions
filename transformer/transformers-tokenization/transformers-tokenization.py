import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        word = (' '.join(texts)).split()
        word = list(Dict.fromkeys(word))
        self.vocab_size = len(word) + 4
        word = sorted(word)
        dict = {}
        dict[self.pad_token] = 0
        dict[self.unk_token] = 1
        dict[self.bos_token] = 2
        dict[self.eos_token] = 3
        for i in range(len(word)):
            dict[word[i]] = i + 4
        self.word_to_id = dict
        for k, v in dict.items():
            self.id_to_word[v] = k
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        text = text.lower()
        word = text.split()
        word = list(Dict.fromkeys(word))
        lst = []
        for i in word:
            if i in self.word_to_id:
                lst.append(self.word_to_id[i])
            else:
                lst.append(self.word_to_id['<UNK>'])
        return lst
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        lst = []
        for i in ids:
            if i in self.id_to_word:
                lst.append(self.id_to_word[i])
            else:
                lst.append('<UNK>')
        return ' '.join(lst)
