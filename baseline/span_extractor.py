from typing import Iterator, Tuple

import spacy


class SpanExtractor:
    skipped_pos = {"ADP", "CCONJ", "SCONJ"}
    allowed_punct = {"\"", "'", "(", ")", "."}

    def __init__(self, model_name: str = "ru_core_news_md"):
        self.nlp = spacy.load(model_name)

    def __call__(self, text: str) -> Iterator[Tuple[int, int]]:
        res = self.nlp(text, disable=("lemmatizer", "ner"))
        for token in res:
            if token.pos_ in {"DET", "PRON"}:
                yield token.idx, token.idx + len(token)
            elif token.pos_ in {"NOUN", "PROPN"}:
                start, end = token.idx, token.idx + len(token)
                for i in range(token.i - 1, -1, -1):
                    if (token.sent != res[i].sent
                            or not token.is_ancestor(res[i])):
                        break
                    if res[i].head == token:
                        if self._is_participle_phrase(res[i]):
                            break
                        if self._is_skipped_pos(res[i]):
                            continue
                        leftmost = next(node for node in res[i].subtree if not self._is_skipped_pos(node))
                        if leftmost.idx < start:
                            start = leftmost.idx
                for i in range(token.i + 1, len(res)):
                    if (token.sent != res[i].sent
                            or not token.is_ancestor(res[i])):
                        break
                    if res[i].head == token:
                        if self._is_participle_phrase(res[i]):
                            break
                        if self._is_skipped_pos(res[i]):
                            continue
                        rightmost = next(node for node in reversed(list(res[i].subtree))
                                         if not self._is_skipped_pos(node))
                        if rightmost.idx + len(rightmost) > end:
                            end = rightmost.idx + len(rightmost)
                yield (start, end)

    @staticmethod
    def _is_participle_phrase(token: spacy.tokens.token.Token) -> bool:
        return token.pos_ == "VERB" and any(node != token for node in token.subtree)

    @staticmethod
    def _is_skipped_pos(token: spacy.tokens.token.Token) -> bool:
        return (token.pos_ in SpanExtractor.skipped_pos
                or (token.pos_ == "PUNCT" and token.text not in SpanExtractor.allowed_punct))
