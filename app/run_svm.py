import joblib
from ml.dataset import Dataset
from ml.svm_classifier import init_model, tag_data, train

svm_model_path = "model_sklearn.pkl"
mode = "train"  # "tagging"
text_corpus = f"ml/data/training_set_clean_only_text.txt"
tag_corpus = f"ml/data/training_set_clean_only_tags.txt"

if __name__ == "__main__":

    if mode == "train":
        dataset = Dataset(text_corpus, tag_corpus, True, True, False)
        acc, f1 = train(dataset, init_model(), svm_model_path)
        print(
            "Model evaluation: \n" f"Model accuracy: {acc:.01%}, " f"Model F1: {f1:.01%}"
        )

    elif mode == "tagging":
        with open(svm_model_path, "rb") as file:
            model = joblib.load(file)
        dataset = Dataset(text_corpus, tag_corpus, False, True, False)
        X = dataset.df["tokens"].values
        tagged = tag_data(X, model)
        tagged = [str(tag) for tag in tagged]
        with open("data/results.txt", "w", encoding="utf-8") as file:  # type:ignore
            file.write("\n".join(tagged))  # type:ignore
