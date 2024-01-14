import json
from argparse import Namespace

from pytest import fixture

from spacy_tokenizer.__main__ import main


@fixture(scope="module")
def dataset() -> list[str]:
    sent = "This is an example sentence, among many other sentences."
    data = [sent] * 10

    return data


@fixture(scope="module")
def txt_dataset(dataset, tmp_path_factory) -> str:
    file = tmp_path_factory.mktemp("dataset") / "dataset.txt"
    file.write_text("\n".join(dataset), encoding="utf-8")

    return str(file)


@fixture(scope="module")
def json_dataset(dataset, tmp_path_factory) -> str:
    file = tmp_path_factory.mktemp("dataset") / "dataset.json"
    file.write_text(json.dumps(dataset), encoding="utf-8")

    return str(file)


@fixture(scope="module")
def default_args() -> Namespace:
    return Namespace(
        input_file=None,  # to override
        output_file=None,  # to override
        spacy_model="en_core_web_sm",
        lowercase=True,
        remove_stopwords=True,
        lemmatize=False,
        remove_punctuation=True,
        verbose=False,
        batch_size=10,
        n_jobs=1,
    )


def test_json_in_json_out(json_dataset, default_args, tmp_path):
    output_file = tmp_path / "output.json"

    default_args.input_file = json_dataset
    default_args.output_file = output_file

    main(default_args)

    assert output_file.is_file()

    data = json.loads(output_file.read_text(encoding="utf-8"))

    assert data[0] == ["example", "sentence", "sentences"]


def test_txt_in_txt_out(txt_dataset, default_args, tmp_path):
    output_file = tmp_path / "output.txt"

    default_args.input_file = txt_dataset
    default_args.output_file = output_file

    main(default_args)

    assert output_file.is_file()

    data = output_file.read_text(encoding="utf-8").splitlines()
    data = data[0].split()

    assert data == ["example", "sentence", "sentences"]
