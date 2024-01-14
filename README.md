# Spacy tokenizer

Spacy-based tokenizer for text documents.


## Installation
```bash
pip install git+https://github.com/adampirog/spacy-tokenizer
```

## Usage

### In code

Constructor arguments:

* **lowercase**: bool = True 
    Convert all characters to lowercase.
* **lemmatize**: bool = False 
    Convert token text to lemmas.
* **remove\_punctuation**: bool = True 
    Remove punctuation and white characters.
* **remove\_stopwords**: bool = True 
    Remove stopwords (the most common words).


### Via script
```bash
spacy-tokenizer # use --help option for details (follows the same API as above)
```
