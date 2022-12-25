import argparse
import json
import os

import torch
from tqdm import tqdm

from doc import Doc
from model import CorefModel
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = CorefModel.load_from_checkpoint(args.weights,).to(args.device).eval()
    tokenizer = utils.load_tokenizer(model.hparams["encoder_model_name"])

    os.makedirs(args.out_dir, exist_ok=True)

    entries = [entry for entry in os.scandir(args.data_dir) if entry.name.endswith(".txt")]
    with torch.no_grad():
        for entry in tqdm(entries, leave=True):
            with open(entry.path, mode="r", encoding="utf8") as f:
                data = {"text": f.read(), "entities": [], "includes": []}
                doc = Doc(entry.name, data, tokenizer=tokenizer, extract_all_spans=True)
                char_entities = [
                    [doc.token_span_to_chars(span) for span in token_entity]
                    for token_entity in model.predict(doc)
                ]
                data["entities"] = char_entities
                data["includes"] = [[] for _ in char_entities]
                out_name = os.path.splitext(entry.name)[0] + ".json"
                with open(os.path.join(args.out_dir, out_name), mode="w", encoding="utf8") as f:
                    json.dump(data, f, ensure_ascii=False)
