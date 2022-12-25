import logging
from typing import Dict, Iterator, List, Optional, Tuple

import transformers

from span_extractor import SpanExtractor


class Doc:

    span_extractor = SpanExtractor()

    def __init__(self,
                 name: str,
                 data: dict,
                 *,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 extract_all_spans: bool = False):
        self.name = name
        self.raw_data = data
        self.encoding = tokenizer(data["text"],
                                  add_special_tokens=True,
                                  padding=True,
                                  truncation=True,
                                  max_length=512,
                                  return_tensors="pt",
                                  return_attention_mask=True,
                                  return_overflowing_tokens=True,
                                  return_offsets_mapping=True,
                                  return_special_tokens_mask=True)
        self.flattening_mask = (self.encoding.special_tokens_mask == 0)
        self.flat_offset_mapping = self.encoding.offset_mapping[self.flattening_mask].tolist()

        self._start_mapping: Dict[int, int] = {}
        self._end_mapping: Dict[int, int] = {}
        for token_i, (char_start, char_end) in enumerate(self.flat_offset_mapping):
            if char_start != char_end:
                if char_start in self._start_mapping or char_end in self._end_mapping:
                    logging.warning(f"{self.name}: "
                                    f"overlapping subtoken {token_i} at {(char_start, char_end)}; "
                                    f"existing subtokens: {self._start_mapping.get(char_start)}, "
                                    f"{self._end_mapping.get(char_end)}")
                    continue
                self._start_mapping[char_start] = token_i
                self._end_mapping[char_end] = token_i

        self.entities = self._match_entities_to_tokens()

        self._all_spans = None
        if extract_all_spans:
            self._all_spans = self._extract_all_spans()

    @property
    def all_spans(self) -> Iterator[Tuple[int, int]]:
        if self._all_spans is None:
            self._all_spans = self._extract_all_spans()
        return iter(self._all_spans)

    def char_span_to_tokens(self, span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        char_start, char_end = span
        token_start = self._start_mapping.get(char_start)
        token_end = self._end_mapping.get(char_end)
        if token_start is not None and token_end is not None:
            return (token_start, token_end + 1)
        logging.warning(f"{self.name}: skipping span {(char_start, char_end)}, "
                        f"{repr(self.raw_data['text'][char_start: char_end])}")

    def token_span_to_chars(self, span: Tuple[int, int]) -> Tuple[int, int]:
        token_start, token_end = span
        char_start = self.flat_offset_mapping[token_start][0]
        char_end = self.flat_offset_mapping[token_end - 1][1]
        return (char_start, char_end)

    def _extract_all_spans(self) -> List[Tuple[int, int]]:
        out = []
        for char_span in self.span_extractor(self.raw_data["text"]):
            token_span = self.char_span_to_tokens(char_span)
            if token_span:
                out.append(token_span)
        return out

    def _match_entities_to_tokens(self) -> List[List[Tuple[int, int]]]:
        char_entities = self.raw_data["entities"]
        token_entities = []

        for char_entity in char_entities:
            token_entity = []
            for char_start, char_end in char_entity:
                token_span = self.char_span_to_tokens((char_start, char_end))
                if token_span:
                    token_entity.append(token_span)
            if token_entity:
                token_entities.append(token_entity)
            else:
                logging.warning(f"{self.name}: skipping entity {char_entity}")

        return token_entities
