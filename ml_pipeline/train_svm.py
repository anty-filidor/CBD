import joblib
from svm.dataset import Dataset
from svm.svm_classifier import init_model, tag_data, train

svm_model_path = "model.pkl"
mode = "train"  # "tagging"
text_corpus = "svm/data/training_set_clean_only_text.txt"
tag_corpus = "svm/data/training_set_clean_only_tags.txt"

if __name__ == "__main__":

    if mode == "train":
        dataset = Dataset(
            texts=text_corpus,
            tags=tag_corpus,
            clean_data=True,
            remove_stopwords=True,
            is_train=True,
        )
        acc, f1 = train(dataset, init_model(), svm_model_path)
        print(
            "Model evaluation: \n" f"Model accuracy: {acc:.01%}, " f"Model F1: {f1:.01%}"
        )

    elif mode == "tagging":
        with open(svm_model_path, "rb") as file:
            model = joblib.load(file)

        with open(text_corpus, encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
        dataset = Dataset(
            texts=lines, clean_data=True, remove_stopwords=True, is_train=False
        )
        X = dataset.df["tokens"].values
        tagged = tag_data(X, model)
        tagged = [str(tag) for tag in tagged]
        with open("svm/data/results.txt", "w", encoding="utf-8") as file:  # type:ignore
            file.write("\n".join(tagged))  # type:ignore
