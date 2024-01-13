import spacy
from pytest import fixture

from spacy_tokenizer import SpacyTokenizer


@fixture(scope="module")
def dataset() -> list[spacy.tokens.Doc]:
    model = spacy.load("en_core_web_sm")
    sent = "This is an example sentence, among many other sentences."
    return model(sent)


def test_spacy_tokenizer(dataset):
    model = SpacyTokenizer(None)

    tokens = model(dataset)[0]
    assert tokens == ["example", "sentence", "sentences"]


def test_input_type(dataset):
    model = SpacyTokenizer(None)

    assert model(dataset) == model([dataset])


def test_string_input():
    model = SpacyTokenizer("en_core_web_sm")
    dataset = "This is an example sentence, among many other sentences."

    tokens = model(dataset)[0]
    assert tokens == ["example", "sentence", "sentences"]


def test_lemmatization(dataset):
    model = SpacyTokenizer(None, lemmatize=True)

    tokens = model(dataset)[0]
    assert tokens == ["example", "sentence", "sentence"]


def test_punctuation(dataset):
    model = SpacyTokenizer(None, remove_punctuation=False)

    tokens = model(dataset)[0]
    assert tokens == ["example", "sentence", ",", "sentences", "."]


def test_stopwords(dataset):
    model = SpacyTokenizer(None, remove_stopwords=False)

    tokens = model(dataset)[0]
    assert tokens == [
        "this",
        "is",
        "an",
        "example",
        "sentence",
        "among",
        "many",
        "other",
        "sentences",
    ]


def test_persistence(tmpdir, dataset):
    model = SpacyTokenizer(None, lemmatize=True)
    before = model(dataset)

    path = tmpdir.mkdir("model")
    path += "/model.json"
    model.save(path)

    model = SpacyTokenizer.load(path)
    after = model(dataset)
    assert before == after
