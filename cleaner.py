# **************************************************************************** #
#                                                                              #
#    cleaner.py                                                                #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/03/24 14:52:35 by Widium                                    #
#    Updated: 2023/03/24 14:52:36 by Widium                                    #
#                                                                              #
# **************************************************************************** #


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re 
import string

# ********************************************************************************************************************#

class Cleaner:
    
    # ********************************************************************************************************************#
    def __init__(
        self,
        string_case : bool = True,
        punctuations : bool = True,
        stop_words : bool = True,
        stem_words : bool = False,
        language : str = "english"
    ) -> None:
        """Cleaning input sentence with multiple string manipulation function

        Args:
            `string_case` (bool, optional): Convert sentence to lower. Defaults to True.
            `punctuations` (bool, optional): remove all punctuation in string.punctuations. Defaults to True.
            `stop_words` (bool, optional): remove stopword. Defaults to True.
            `stem_words` (bool, optional): stemming words. Defaults to False.
        """
        self.case = string_case
        self.punctuations = punctuations
        self.stop_words = stop_words
        self.stem_words = stem_words

        self.porter = PorterStemmer()
        self.language = language

    # ********************************************************************************************************************#
    
    def case_cleaner(self,
                     sentence : str)->str:
    
        sentence_lower = sentence.lower()
        
        return (sentence_lower)
    
    # ********************************************************************************************************************#
    
    def punctuations_cleaner(self,
                            sentence : str)->str:
        
        all_punctation = re.escape(string.punctuation)
        cleaner = re.compile(f"[{all_punctation}]")
        
        words = cleaner.sub(repl="", string=sentence)
        
        words_without_space = words.split()
        
        clean_sentence = " ".join(words_without_space)
        
        return (clean_sentence)

    # ********************************************************************************************************************#
    
    def stop_words_cleaner(self,
                           sentence : str)->str:
        
        stop_words = stopwords.words(self.language)
        words = sentence.split(sep=" ")
        
        stop_words_removed = [
            word for word in words
            if not word in stop_words
        ]
        
        cleaned_sentence = " ".join(stop_words_removed)
        
        return (cleaned_sentence)

    # ********************************************************************************************************************#
    
    def stem_words_converter(self,
                             sentence : str)->str:
        
        words = sentence.split(sep=" ")
        words_stemmed = [
            self.porter.stem(word)
            for word in words
        ]
        
        cleaned_sentence = " ".join(words_stemmed)
        
        return (cleaned_sentence)

    # ********************************************************************************************************************#
    
    def apply(self, sentence : str)->str:
        
        if self.case:
            sentence = self.case_cleaner(sentence)
        if self.punctuations:
            sentence = self.punctuations_cleaner(sentence)
        if self.stop_words:
            sentence = self.stop_words_cleaner(sentence)
        if self.stem_words:
            sentence = self.stem_words_converter(sentence)

        return (sentence)
    # ********************************************************************************************************************#