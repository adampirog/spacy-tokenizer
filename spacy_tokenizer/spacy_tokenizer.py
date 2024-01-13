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
        if spacy_model:
            self.spacy_model = spacy.load(spacy_model)
        else:
            self.spacy_model = None

        self.spacy_model_name = spacy_model
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize

    def __call__(self, *args, **kwargs) -> OutputType:
        return self.transform(*args, **kwargs)

    def transform(
        self,
        texts: InputType,
        *,
        batch_size: int | None = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> OutputType:
        if isinstance(texts, (Doc, str)):
            texts = [texts]

        if not isinstance(texts[0], Doc):
            texts = tqdm(
                self.spacy_model.pipe(
                    texts,
                    batch_size=batch_size,
                    n_process=n_jobs,
                ),
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
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)

        return cls(**config)
