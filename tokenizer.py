# **************************************************************************** #
#                                                                              #
#    tokenizer.py                                                              #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/03/24 14:53:12 by Widium                                    #
#    Updated: 2023/03/24 16:32:34 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from typing import List, Dict

# ********************************************************************************************************************#

class Tokenizer:
    """Mapping Words in Entire Dataset in Vocabulary
       Encode Sentence and Decode Sequence of Tokens
    """
# ********************************************************************************************************************#
    
    def __init__(
        self,
        dataset : List[str],
        max_vocab_size : int,
    )-> None:
        """Initialize Instance with Dataset of Cleaned Sentence

        Args:
            `dataset` (List[str]): list of Cleaned sentence
            `max_vocab_size` (int): size maximum of Vocabulary
        """
        
        self.sentences = dataset
        self.word_vocabulary = dict()
        self.token_vocabulary = dict()
        self.max_vocab_size = max_vocab_size
        self.padding_token = None

# ********************************************************************************************************************#   
    
    def create_vocabulary(self)->None:
        """Create Vocabulary and Reverse Vocabulary
           Concatenate all sentences in list of words
           map unique words in both dictionary
        """
        
        concatenate_sentences = " ".join(self.sentences)
        split_words = concatenate_sentences.split(sep=" ")
        split_words += ["<start>", "<pad>", "<stop>", "<unk>"]
        
        for word in split_words:
            vocab_size = len(self.word_vocabulary)
            
            if vocab_size <= self.max_vocab_size:
                
                if word not in self.word_vocabulary:
                    index = vocab_size
                    self.word_vocabulary[word] = index
                    self.token_vocabulary[index] = word
        
        self.padding_token = self.word_vocabulary["<pad>"]

# ********************************************************************************************************************#
        
    def encode_sentence(self,
                        sentence : str)->List[int]:
        """Use the Word Vocabulary for Encode each word in Sentence 
        
        add special tokens `<start>` `<stop>` in sentence before encoding
        convert each word to token, when word is not found add `<unk>` token instead

        Args:
            `sentence` (str): input sentence

        Returns:
            `List[int]`: list of token
        """
        encoded_sentence = list()
        
        sentence = f"<start> {sentence} <stop>"
        
        split_words = sentence.split(sep=" ")
        
        for word in split_words:
            
            if word in self.word_vocabulary:
                token = self.word_vocabulary[word]
            else :
                token = self.word_vocabulary["<unk>"]
            
            encoded_sentence.append(token)
        
        return (encoded_sentence)

# ********************************************************************************************************************#

    def decode_tokens(self,
                        tokens : List[int])->str:
        """Use the Token Vocabulary to Decode the Sequence of tokens

        Args:
            `tokens` (List[int]): sequence of tokens

        Returns:
            `str`: string with decoded tokens
        """
        sentence = [
            self.token_vocabulary[token]
            for token in tokens
        ]
        return (sentence)

# ********************************************************************************************************************#