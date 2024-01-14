"""
Script for SpacyTokenizer documents processing.
"""
import json
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path

from .spacy_tokenizer import SpacyTokenizer


def read_file(path: str) -> list[str]:
    path = Path(path)

    if path.suffix == ".txt":
        data = path.read_text(encoding="utf-8").splitlines()
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        raise ValueError(f"Format '{path.suffix}' not supported.")

    return data


def write_file(data: list[list[str]], path: str) -> None:
    path = Path(path)

    if path.suffix == ".txt":
        with path.open("wt", encoding="utf-8") as handle:
            for line in data:
                for word in line:
                    handle.write(word + " ")
                handle.write("\n")
    elif path.suffix == ".json":
        with path.open("wt", encoding="utf-8") as handle:
            json.dump(data, handle)
    else:
        raise ValueError(f"Format '{path.suffix}' not supported.")


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input file to tokenize. "
        "For .txt format - provide each document in seperate line. "
        "For .json format - provide an array of documents.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="File to store the results (json or txt format).",
    )
    parser.add_argument(
        "spacy_model", type=str, help="Name of the spacy model."
    )
    parser.add_argument(
        "-l",
        "--lowercase",
        default=False,
        action="store_true",
        help="Convert all characters to lowercase.",
    )
    parser.add_argument(
        "-s",
        "--remove-stopwords",
        default=False,
        action="store_true",
        help="Remove stopwords (the most common words)",
    )
    parser.add_argument(
        "-p",
        "--remove-punctuation",
        default=False,
        action="store_true",
        help="Remove punctuation and white characters.",
    )
    parser.add_argument(
        "--lemmatize",
        default=False,
        action="store_true",
        help="Convert token text to lemmas.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Display progress bar.",
    )
    parser.add_argument(
        "-n",
        "--n-jobs",
        type=int,
        default=1,
        help="Number of concurrent processes for spacy model processing. "
        "(default: 1)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for spacy processing "
        "(default: 'spacy_model.batch_size')",
    )

    return parser.parse_args()


def main(args: Namespace):
    data = read_file(args.input_file)

    tokenizer = SpacyTokenizer(
        spacy_model=args.spacy_model,
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        remove_stopwords=args.remove_stopwords,
        lemmatize=args.lemmatize,
    )

    result = tokenizer.transform(
        data,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    write_file(result, args.output_file)


def cli():
    main(parse_args())


if __name__ == "__main__":
    cli()
