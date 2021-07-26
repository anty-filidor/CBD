import time
from glob import glob
from typing import List, Tuple

import pandas as pd
from dataset import Dataset
from googletrans import Translator

dataset_dir = "data"
dataset_temp_dir = f"{dataset_dir}/enhanced_dataset"

sub_corpus_name = "metadata.csv"
text_corpus_path = f"{dataset_dir}/training_set_clean_only_text.txt"
tag_corpus_path = f"{dataset_dir}/training_set_clean_only_tags.txt"

enhanced_text_corpus_path = f"{dataset_dir}/training_set_texts.txt"
enhanced_tag_corpus_path = f"{dataset_dir}/training_set_tags.txt"


def extract_1_2_classes_from_corpus() -> None:
    """Creates sub corpus that contains only cyberbullying and hate-speech classes."""
    dataset = Dataset(text_corpus_path, tag_corpus_path, True, True, False)

    new_dataset = dataset.df.copy()
    new_dataset = new_dataset.loc[new_dataset["tag"] != 0].reset_index()
    new_dataset = new_dataset[["tag", "text"]]
    new_dataset = new_dataset.astype({"tag": str, "text": str})

    new_dataset.to_csv(sub_corpus_name)


def enhance_corpus(batch_num: int) -> None:
    """
    Translates selected batch of sub_dataset in way: pl -> en -> de -> pl.

    Due to limitations of Google Translate API we were forced not to process entire
    dataset but only batch of size 80 records.

    :param batch_num: number of batch to be processed
    """
    ddf = pd.read_csv(sub_corpus_name)

    min_limit = 80 * batch_num
    max_limit = 80 * (batch_num + 1)
    if max_limit > len(ddf):
        min_limit = len(ddf)

    print(f"Iteration {batch_num}, min_limit: {min_limit}, max_limit: {max_limit}")

    ddf = pd.read_csv("metadata.csv")
    ddf = ddf.astype({"tag": str, "text": str})
    ddf = ddf.loc[min_limit:max_limit]
    my_texts = ddf["text"].to_list()
    my_tags = ddf["tag"].to_list()

    tags_path = (
        f"{dataset_temp_dir}/training_set_clean_only_tags_enhanced_{batch_num}.txt"
    )
    text_path = (
        f"{dataset_temp_dir}/training_set_clean_only_text_enhanced_{batch_num}.txt"
    )

    with open(tags_path, "w", encoding="utf-8") as file:
        tt = "\n".join(my_tags)
        file.write(tt + "\n")

    translator = Translator()

    with open(text_path, "w", encoding="utf-8") as file:
        for text in my_texts:
            pl_en = translator.translate(text, dest="en", src="pl").text
            pl_en_de = translator.translate(pl_en, dest="de", src="en").text
            pl_en_de_pl = translator.translate(pl_en_de, dest="pl", src="de").text
            file.write(f"{pl_en_de_pl}\n")


def _merge_batches(batch_type: str) -> Tuple[List[str], int, int]:
    """Merges all batches of 'text' or 'tags' to one file."""
    batches = glob(f"{dataset_temp_dir}/*{batch_type}*.txt")

    merged_list = []
    for batch in batches:
        with open(batch, "r", encoding="utf-8") as f_batch:
            t = f_batch.readlines()
            merged_list += t

    return merged_list, len(batches), len(merged_list)


def merge_translated_corpus():
    """Merges created enhanced corpus into one additional file and validates results."""
    enhanced_tags, tags_len_batches, tags_len_records = _merge_batches("tags")
    enhanced_texts, text_len_batches, text_len_records = _merge_batches("text")

    assert text_len_records == tags_len_records
    assert text_len_batches == tags_len_batches

    with open(tag_corpus_path, "r", encoding="utf-8") as f_batch:
        original_tags = f_batch.readlines()
    original_tags += enhanced_tags
    with open(enhanced_tag_corpus_path, "w", encoding="utf-8") as file:
        file.write("".join(original_tags))

    with open(text_corpus_path, "r", encoding="utf-8") as f_batch:
        original_texts = f_batch.readlines()
    original_texts += enhanced_texts
    with open(enhanced_text_corpus_path, "w", encoding="utf-8") as file:
        file.write("".join(original_texts))


if __name__ == "__main__":

    extract_1_2_classes_from_corpus()

    for batch in range(0, 10):
        enhance_corpus(batch)
        time.sleep(10)

    merge_translated_corpus()
