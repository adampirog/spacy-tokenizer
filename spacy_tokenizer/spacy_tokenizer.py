import json
from typing import Optional, Self

import spacy
from spacy.tokens import Doc
from tqdm.auto import tqdm


InputType = Doc | str | list[str] | list[Doc]
OutputType = list[list[str]]


class SpacyTokenizer:
    def __init__(
        self,
        spacy_model: Optional[str],
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = False,
    ) -> None:
        """
        Initialize a tokenizer based on a spacy language model.

        Parameters
        ----------

        spacy_model : Optional[str]
            Name of the spacy model to use. You can provide None to skip
            the loading of spacy model - then you must provide input data
            in spacy.Doc format.

        lowercase : bool = True
            Convert all characters to lowercase

        remove_punctuation: bool = True
            Remove punctuation and white characters

        remove_stopwords : bool = True
            Remove stopwords (the most common words)

        lemmatize : bool = False
            Convert token text to lemmas
        """

        if spacy_model:
            self.spacy_model = spacy.load(spacy_model)
        else:
            self.spacy_model = None

        self.spacy_model_name = spacy_model
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize

    def __call__(
        self,
        texts: InputType,
        *,
        batch_size: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> OutputType:
        """
        Perform the tokenization

        If tokenizer has not initialized spacy model, you must provide input data
        in spacy.Doc format.
        """

        return self.transform(
            texts=texts,
            batch_size=batch_size,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def transform(
        self,
        texts: InputType,
        *,
        batch_size: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> OutputType:
        """
        Perform the tokenization

        If tokenizer has not initialized spacy model, you must provide input data
        in spacy.Doc format.
        """

        if isinstance(texts, (Doc, str)):
            texts = [texts]

        if not isinstance(texts[0], Doc):
            pipe_args = {"n_process": n_jobs}
            if batch_size:
                pipe_args["batch_size"] = batch_size

            texts = tqdm(
                self.spacy_model.pipe(texts, **pipe_args),
                disable=not verbose,
                total=len(texts),
                desc="Tokenizing",
            )

        return [self._tokenize(doc) for doc in texts]

    def _tokenize(self, doc: Doc) -> list[str]:
        result = []
        for token in doc:
            if self.remove_punctuation and (token.is_punct or token.is_space):
                continue
            if self.remove_stopwords and token.is_stop:
                continue

            if self.lemmatize:
                text = token.lemma_
            else:
                text = token.text

            if self.lowercase:
                text = text.lower()

            result.append(text)

        return result

    def save(self, path: str) -> None:
        """
        Save the tokenizer

        (as a json file)
        """

        config = {
            "spacy_model": self.spacy_model_name,
            "lowercase": self.lowercase,
            "remove_stopwords": self.remove_stopwords,
            "remove_punctuation": self.remove_punctuation,
            "lemmatize": self.lemmatize,
        }

        with open(path, "wt", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4)

    @classmethod
    def load(cls, path) -> Self:
        """
        Load saved tokenizer

        (from a json file)
        """

        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)

        return cls(**config)
