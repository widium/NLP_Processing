# **************************************************************************** #
#                                                                              #
#    language_dataset.py                                                       #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/03/24 16:57:14 by Widium                                    #
#    Updated: 2023/03/24 16:57:16 by Widium                                    #
#                                                                              #
# **************************************************************************** #


from typing import List, Dict

from torch.utils.data import Dataset
from torch import Tensor
import torch

from cleaner import Cleaner
from tokenizer import Tokenizer
from vectorizer import Vectorizer

# ********************************************************************************************************************#

class LanguageModelingDataset(Dataset):
    """Create Dataset with Sentence Cleaned, Tokenized and Vectorized 
       For Language Modeling 

    Args:
        Dataset (Pytorch Dataset): Subclassing the `Dataset` class
    """
# ********************************************************************************************************************#

    def __init__(
        self,
        dataset: List[Dict],
        vocab_size : int = 10_000,
        max_tokens : int = 20,
        string_case : bool = True,
        punctuations : bool = True,
        stop_words : bool = True,
        stem_words : bool = False,
        language : str = "english"
    )->None:
        """Initialize Dataset with Preprocessing NLP Class
        
        Clean all text 
        Create Vocabulary 
        Initialize Vectorizer

        Args:
            dataset (List[Dict]): Dataset with List of Dictionary with sentence as value like `dataset = {"text" : "Hi everyone"}`
            vocab_size (int, optional): Vocabulary Size. Defaults to 10_000.
            max_tokens (int, optional): Max tokens per Vector. Defaults to 20.
            string_case (bool, optional): Apply removing string case. Defaults to True.
            punctuations (bool, optional): Apply removing punctuations. Defaults to True.
            stop_words (bool, optional): Apply removing stop words. Defaults to True.
            stem_words (bool, optional): Apply convert stem words. Defaults to False.
            language (str, optional): choose language for stem words and stop words. Defaults to "english".
        """
        self.sentences = [sample["text"] for sample in dataset]
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        
        self.cleaner = Cleaner(
            string_case,
            punctuations,
            stop_words,
            stem_words,
            language
        )
        
        self.cleaned_dataset = [
            self.cleaner.apply(sentence=sample)
            for sample in self.sentences
        ]
        
        self.tokenizer = Tokenizer(
            max_vocab_size=self.vocab_size,
            dataset=self.cleaned_dataset)
        
        self.tokenizer.create_vocabulary()
        
        self.vectorizer = Vectorizer(
            max_tokens=self.max_tokens,
            padding_token=self.tokenizer.padding_token
        )

        self.word_vocabulary = self.tokenizer.word_vocabulary
        self.token_vocabulary = self.tokenizer.token_vocabulary

# ********************************************************************************************************************#

    def __len__(self)->int:
        """Return Number of Sentence in Dataset"""
        return (len(self.sentences))

# ********************************************************************************************************************#
 
    def __getitem__(self, idx : int)->Tensor:
        """When use `[]` with index on instance Preprocessing sample like
        
        Cleaning
        Tokenizing
        Vectorizing

        Args:
            idx (int): index of sample

        Returns:
            Tensor: 1D Vector of Tokens ID
        """
        sentence = self.sentences[idx]
        cleaned_sentence = self.cleaner.apply(sentence=sentence)
        tokens = self.tokenizer.encode_sentence(sentence=cleaned_sentence)
        vector = self.vectorizer.vectorize_tokens(tokens=tokens)
        
        return (torch.tensor(vector))
    
# ********************************************************************************************************************#