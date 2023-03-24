# **************************************************************************** #
#                                                                              #
#    vectorizer.py                                                             #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/03/24 14:53:39 by Widium                                    #
#    Updated: 2023/03/24 15:28:15 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import numpy as np
from typing import List

# ********************************************************************************************************************#

class Vectorizer:
    """Convert list of Tokens ID to Numpy Vector with fixed lenght"""

# ********************************************************************************************************************#
   
    def __init__(self,
                 max_tokens : int,
                 padding_token : int) -> None:
        """Initialize Instance with :
            - fixed number of tokens
            - padding token ID  
           
        Args:
            `max_sequence_lenght` (int): max lenght of vector
            `padding_token` (int): padding token ID  
        """
        self.padding_token = padding_token
        self.max_tokens = max_tokens

# ********************************************************************************************************************# 
  
    def vectorize_tokens(self,
                        tokens : List[int])->np.array:
        """Use the Tokenizer for Encode Sentence in Fixed Lenght
        
        id the vector is bigger than `self.max_sequence_lenght`
        the vector going to be cut
        if the vector is small than `self.max_sequence_lenght`
        add padding tokens for return the same size of vector

        Args:
            `tokens` (List[int]): Input list of Tokens ID

        Returns:
            `np.array`: 1D Vector with tokens ID
        """
        vector = tokens[:self.max_tokens]
        
        if len(vector) < self.max_tokens:
            nbr_padding = self.max_tokens - len(vector)
            vector = vector + ([self.padding_token] * nbr_padding)
        
        return (np.array(vector))

# ********************************************************************************************************************#