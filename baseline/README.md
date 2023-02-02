This is a simple neural baseline model which is only capable of predicting identity-coreference links, but is not able to predict links to split antecedents. \
The default model uses `sberbank-ai/ruRoberta-large` as the encoder.

The architecture is loosely based on these two papers: [Lee et al., 2018](https://aclanthology.org/N18-2108); [Joshi et al., 2020](https://aclanthology.org/D19-1588). The main differences are in dropping the higher-order iterative approach (following [Xu et al., 2020](https://aclanthology.org/2020.emnlp-main.686.pdf)), simpler span encoding (following [Dobrovolskii, 2021](https://aclanthology.org/2021.emnlp-main.605)) and using rule-based span extraction.

Tested on Python 3.9.15.

Install the dependencies:
```
python -m pip install -r requirements.txt
python -m spacy download ru_core_news_md
```
Download and extract the training data:
```
wget https://github.com/vdobrovolskii/rucoco/releases/download/v1.0.0/v1.0.0.zip
unzip v1.0.0.zip -d data
```
## Training
Split data to train/val:
```
python split_data.py
```
This will produce the following structure:
```
.
└── split_data
    ├── train
    └── val
```
Train the model with default settings:
```
python train.py --data_dir split_data --accelerator gpu --devices 1
```
The `--max_batches_train` parameter can help save GPU memory. For 12 GB GPU it is recommended to set it to `1`, for 16 GB it can be `2`. \
See the list of all training parameters by running:
```
python train.py --help
```
## Prediction
[Download](https://abbyyihq-my.sharepoint.com/:u:/r/personal/vladimir_dobrovolskiy_abbyy_com/Documents/baseline.ckpt?csf=1&web=1&e=iJxDwN) the pretrained weights. Then predict:
```
python predict.py TXT_DATA_DIR OUT_DIR --weights PATH_TO_CKPT --device cuda:0
```
For example:
```
python predict.py test test_out --weights baseline.ckpt --device cuda:0
```
