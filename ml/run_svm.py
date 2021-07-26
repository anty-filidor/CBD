import argparse

import joblib
from dataset import Dataset
from models.svm_classifier import init_model, tag_data, train

svm_model_path = "data/svm-data/model/model_independent.pkl"


def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--text_corpus",
        type=str,
        required=True,
        help="Path to twitter messages in txt file",
    )
    parent_parser.add_argument(
        "--tag_corpus", type=str, required=True, help="Path to labels in txt file"
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # subparser for training
    subparsers.add_parser("train", parents=[parent_parser])

    # subparser for tagging
    subparsers.add_parser("tagging", parents=[parent_parser])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        dataset = Dataset(args.text_corpus, args.tag_corpus, True, True, False)
        acc, f1 = train(dataset, init_model(), svm_model_path)
        print(
            "Model evaluation: \n" f"Model accuracy: {acc:.01%}, " f"Model F1: {f1:.01%}"
        )

    elif args.mode == "tagging":
        with open(svm_model_path, "rb") as file:
            model = joblib.load(file)
        dataset = Dataset(args.text_corpus, args.tag_corpus, False, True, False)
        X = dataset.df["tokens"].values
        tagged = tag_data(X, model)
        tagged = [str(tag) for tag in tagged]
        with open("data/results.txt", "w", encoding="utf-8") as file:  # type:ignore
            file.write("\n".join(tagged))  # type:ignore
