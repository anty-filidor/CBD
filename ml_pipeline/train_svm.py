import joblib
from svm.dataset import Dataset
from svm.svm_classifier import init_model, tag_data, train

svm_model_path = "model.pkl"


def _get_train_params():
    return "train", "svm/data/training_set_texts.txt", "svm/data/training_set_tags.txt"


def _get_validat_params():
    return "tag", "svm/data/test_set_only_text.txt", "svm/data/test_set_only_tags.txt"


# mode, text_corpus, tag_corpus = get_train_params()
mode, text_corpus, tag_corpus = _get_validat_params()


if __name__ == "__main__":

    if mode == "train":
        dataset = Dataset(
            texts=text_corpus,
            tags=tag_corpus,
            clean_data=True,
            remove_stopwords=True,
            is_train=True,
        )
        dataset.print_stats()
        acc, f1 = train(dataset, init_model(), svm_model_path)
        print(
            "Model evaluation: \n" f"Model accuracy: {acc:.01%}, " f"Model F1: {f1:.01%}"
        )

    elif mode == "tag":
        with open(svm_model_path, "rb") as file:
            model = joblib.load(file)

        with open(text_corpus, encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]

        dataset = Dataset(
            texts=lines,
            clean_data=False,
            remove_stopwords=True,
            is_train=False,
        )
        X = dataset.df["tokens"].values
        tagged = tag_data(X, model)
        tagged = [str(tag) for tag in tagged]
        with open("svm/data/results.txt", "w", encoding="utf-8") as file:  # type:ignore
            file.write("\n".join(tagged))  # type:ignore
