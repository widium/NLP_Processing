# NLP Processing
Create Cleaner, Tokenizer and Vectorizer From Scratch for Learning Purpose

![](https://i.imgur.com/J5zDGMD.png)
***
```python
dataset = [
    {'text': "the game began development in 2010 ..."},
    {'text': "it met with positive sales in japan , ..."},
    {'text': "troops are divided into five classes : ..."},
    {'text': "unlike its two predecessors , valkyria chroni..."},
    {'text': 'on its day of release in japan , valkyria chr...'},
    {'text': 'famitsu enjoyed the story , and wer....'},
    {'text': "in a preview of the tgs demo ,..."},
    {'text': 'kurt and <unk> were featured in the ni..'},
    {'text': 'the building receives its name ...'},
    {'text': 'the united states troops at the ou...'}
]
```


```python
dataset[0]["text"]
```
    "the game began development in 2010..."



## [Cleaner](/cleaner.py)
setup specified task of `Cleaner` like :
- `string_case` : `"HeLlO"` to `"hello"`
```python
"HeLlO" to "hello"
```
- `punctuations` : `"?m/:y;# n'ame i(s E{d}die` to `"my name is Eddie"`
```python
"?m/:y;# n'ame i(s E{d}die" to "my name is Eddie"
```
- `stop_words` : `"How do i install a software on my Computer"` to `"How install software Computer"`
```python
"How do i install a software on my Computer" to "How install software Computer"
```
- `stem_words` : 
```python
"Im a Machine Learning Engineer who love Programming in Object Oriented Programming"
to 
"im a machin learn engin who love program in object orient program"
```

```python
from cleaner import Cleaner

cleaner = Cleaner(
    string_case=True,
    punctuations=True,
    stop_words=False,
    stem_words=False,
)
original_sentence = "The 'Game' began development in 2010 , carrying over a large portion of the work done on valkyria chronicles..."
cleaned_sentence = cleaner.apply(sentence=original_sentence)

print(
    f"Original : {original_sentence}\n",
    f"Cleaned :{cleaned_sentence}"
)
```
```python
>>> Original : The 'Game' began development in 2010 , carrying over a large portion of the work done on valkyria chronicles...

>>> Cleaned :the game began development in 2010 carrying over a large portion of the work done on valkyria chronicles
```


```python
dataset

>>> {'text': "the game began development in 2010 ..."},
{'text': "it met with positive sales in japan , ..."},
{'text': "troops are divided into five classes : ..."},
{'text': "unlike its two predecessors , valkyria chroni..."},
{'text': 'on its day of release in japan , valkyria chr...'},
{'text': 'famitsu enjoyed the story , and wer....'},
{'text': "in a preview of the tgs demo ,..."},
{'text': 'kurt and <unk> were featured in the ni..'},
{'text': 'the building receives its name ...'},
{'text': 'the united states troops at the ou...'}
```

```python
cleaned_dataset = [
    cleaner.apply(sentence=sample["text"])
    for sample in dataset
]
```
```python
cleaned_dataset

>>> ['the game began development in 2010 carrying over ..',
'it met with positive sales in japan and was praised..',
'troops are divided into five classes scouts unk engin..',
'unlike its two predecessors valkyria chronicles iii w...4',
'on its day of release in japan valkyria chronicles iii t...',
'famitsu enjoyed the story and were particularly pleased with ...',
'in a preview of the tgs demo ryan geddes of ign was left ...',
'kurt and unk were featured in the nintendo ...',
'the building receives its name from its distinct ...',
'the united states troops at the outposts of...']
```

## [Tokenizer](/tokenizer.py)
Create Vocabulary of Words on Multi Cleaned Sentence
- `word_vocabulary` : dictionary with `word : tokenID` for encode sentence
- `token_vocabulary` : dictionary with `tokenID : word` for decode tokens
- like :
```python
word_vocabulary
>>> 
{'the': 0,
 'game': 1,
 'began': 2,
 'development': 3,
 'in': 4,
 '2010': 5,
 'carrying': 6,
 'over': 7,
 'a': 8
 ...}

token_vocabulary
>>>
{0: 'the',
 1: 'game',
 2: 'began',
 3: 'development',
 4: 'in',
 5: '2010',
 6: 'carrying',
 7: 'over',
 8: 'a',
...'}
```
- encode sentence with `word_vocabulary` and some special tokens `<start>` `<stop>`, `<unk>`, ...
- decode tokens list with `token_vocabulary`


```python
from tokenizer import Tokenizer

tokenizer = Tokenizer(max_vocab_size=1000,
                      dataset=cleaned_dataset)

tokenizer.create_vocabulary()

print(
    len(tokenizer.word_vocabulary), 
    len(tokenizer.token_vocabulary)
)

tokens = tokenizer.encode_sentence(sentence="the game starting in 2010")
sentence = tokenizer.decode_tokens(tokens=tokens)

print(f"Tokenized Sentence : {tokens}")
print(f"Decoded tokens : {sentence}")
```
```python
>>> 367 367
>>> Tokenized Sentence : [363, 0, 1, 325, 4, 5, 365]
>>> Decoded tokens : ['<start>', 'the', 'game', 'starting', 'in', '2010', '<stop>']
```
## [Vectorizer](/vectorizer.py) 
- the vectorizer convert list of Tokens ID to 1 dimensional vector with fixed lenght
- the fixed lenght (`max_sequence_lenght`) is used for return the same size of vector for each sentence
- if the list of tokens is smaller than vector lenght, add padding tokens `<pad>`
- if the list of tokens is bigger thant vector lenght, cut the tokens list with indexing


```python
from vectorizer import Vectorizer

vectorizer = Vectorizer(
    max_tokens=20,
    padding_token=tokenizer.padding_token
)

tokens = tokenizer.encode_sentence(sentence="the game starting in 2010")
vector = vectorizer.vectorize_tokens(tokens=tokens)
sentence = tokenizer.decode_tokens(vector)

print(f"--- Tokenized Sentence ---\n{tokens}\n")
print(f"--- Vectorized Sentence ---\nShape : {vector.shape}\nVector : {vector}\n")
print(f"--- Decoded tokens ---\nSentence : {sentence}")
```
```python
--- Tokenized Sentence ---
[363, 0, 1, 325, 4, 5, 365]

--- Vectorized Sentence ---
Shape : (20,)
Vector : [363 0 1 325 4 5 365 364 364 364 364 364 364 364 364 364 364 364 364 364]

--- Decoded tokens ---
Sentence : ['<start>', 'the', 'game', 'starting', 'in', '2010', '<stop>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
```

## [LanguageModelingDataset](/language_dataset.py)
- Create Dataset with all Preprocessing Class together
- Cleaning
- Tokenizing
- Vectorizing
- When call random index in instance it return Vectorized sample

```python
import random
from language_dataset import LanguageModelingDataset

VOCAB_SIZE = 10_000
MAX_TOKENS = 20

language_dataset = LanguageModelingDataset(
    dataset=dataset,
    vocab_size=VOCAB_SIZE,
    max_tokens=MAX_TOKENS,
    string_case=True,
    punctuations=True,
    stop_words=True,
    stem_words=False,
    language="english"
)

random_index = random.randint(0, len(language_dataset) - 1)

print(
    f"--- Size of Dataset ---\n{len(language_dataset)}\n",
    f"--- Random Sample ---\n{language_dataset[random_index]}"
)
```
```python
--- Size of Dataset ---
10
    --- Random Sample ---
tensor([313, 289, 290,  82, 291,  51, 292, 288, 293, 294, 295, 296, 297, 298,
        299, 300, 301, 299, 300, 301])
```
