import argparse
from copy import deepcopy
import json
import os

import pytorch_lightning as pl
import torch
import transformers

from doc import Doc
from model import CorefModel
import utils


class CorefDocs(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizerFast):
        self.docs = []
        for entry in os.scandir(path):
            if entry.name.endswith(".json"):
                with open(entry.path, mode="r", encoding="utf8") as f:
                    data = json.load(f)
                    self.docs.append(Doc(entry.name, data, tokenizer=tokenizer, extract_all_spans=True))

    def __getitem__(self, idx: int) -> Doc:
        return self.docs[idx]

    def __len__(self) -> int:
        return len(self.docs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="split_data")

    group = parser.add_argument_group("EarlyStopping")
    group.add_argument("--patience", type=int, default=10)

    group = parser.add_argument_group("TrainDataLoader")
    group.add_argument("--num_workers", type=int, default=4)

    parser = CorefModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()

    tokenizer = utils.load_tokenizer(args.encoder_model_name)
    model = CorefModel(**vars(args))

    train_data = CorefDocs(os.path.join(args.data_dir, "train"), tokenizer)
    val_data = CorefDocs(os.path.join(args.data_dir, "val"), tokenizer)

    collate_fn = lambda x: deepcopy(x)
    train_data_loader = torch.utils.data.DataLoader(train_data, shuffle=True, collate_fn=collate_fn,
                                                    num_workers=args.num_workers)
    val_data_loader = torch.utils.data.DataLoader(val_data, collate_fn=collate_fn)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor="val_lea", min_delta=0.01, mode="max",
                                                               patience=args.patience)
    checkpointing = pl.callbacks.ModelCheckpoint(monitor="val_lea", mode="max", filename="{epoch:02d}-{val_lea:.3f}")
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[early_stopping, checkpointing])

    trainer.fit(model, train_data_loader, val_data_loader)
