from tokenizers import pre_tokenizers
import transformers


ADD_PUNCTUATION_PRE_TOKENIZER = {"sberbank-ai/ruRoberta-large", }


def load_tokenizer(model_name: str) -> transformers.AutoTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if model_name in ADD_PUNCTUATION_PRE_TOKENIZER:
        tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(),
                                                                      tokenizer._tokenizer.pre_tokenizer])
    return tokenizer
