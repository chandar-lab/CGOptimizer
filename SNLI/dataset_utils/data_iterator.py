import torch
from torchtext import data
from torchtext.data import BucketIterator
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def DataIteratorGlove(
    data_folder="./datasets/", dataset="snli", batch_size=2, max_length=200, prefix=""
):
    TEXT = data.Field(
        fix_length=None,
        eos_token="</s>",
        pad_token="<pad>",
        init_token="<s>",
        unk_token="<unk>",
        include_lengths=True,
    )
    label = data.LabelField()
    train_dialog = data.TabularDataset(
        path=os.path.join(data_folder, dataset, prefix + dataset + "_train_glove.csv"),
        format="csv",
        fields=[
            ("context_id", None),
            ("Sentence1", TEXT),
            ("Sentence2", TEXT),
            ("Label", label),
        ],
    )
    valid_dialog = data.TabularDataset(
        path=os.path.join(data_folder, dataset, prefix + dataset + "_dev_glove.csv"),
        format="csv",
        fields=[
            ("context_id", None),
            ("Sentence1", TEXT),
            ("Sentence2", TEXT),
            ("Label", label),
        ],
    )
    test_dialog = data.TabularDataset(
        path=os.path.join(data_folder, dataset, prefix + dataset + "_test_glove.csv"),
        format="csv",
        fields=[
            ("context_id", None),
            ("Sentence1", TEXT),
            ("Sentence2", TEXT),
            ("Label", label),
        ],
    )

    TEXT.build_vocab(train_dialog)
    label.build_vocab(train_dialog)

    label.vocab.stoi = defaultdict(None)
    label.vocab.stoi["entailment"] = 2
    label.vocab.stoi["neutral"] = 1
    label.vocab.stoi["contradiction"] = 0
    label.vocab.itos = ["contradiction", "neutral", "entailment"]

    train_dialog_iter = BucketIterator.splits(
        (train_dialog),
        batch_size=batch_size,
        sort_key=lambda x: len(x.Label),
        sort_within_batch=False,
        device=device,
    )

    # return train_dialog_iter, valid_dialog_iter, test_dialog_iter, TEXT.vocab.stoi['<pad>'], len(TEXT.vocab), TEXT.vocab.itos#TEXT.vocab
    return (
        train_dialog_iter,
        None,
        None,
        TEXT.vocab,
        label.vocab,
    )
