from typing import List, NamedTuple, Optional, Tuple

import pytorch_lightning as pl
import torch
from transformers import AutoModel

from doc import Doc


EPS = 1e-8


class LEAResult(NamedTuple):
    precision: float
    precision_weight: float
    recall: float
    recall_weight: float


class CorefModel(pl.LightningModule):
    def __init__(self,
                 encoder_model_name: str = "sberbank-ai/ruRoberta-large",
                 dropout_rate: float = 0.3,
                 k: int = 50,
                 max_batches_train: Optional[int] = None,
                 **discarded_kwargs):
        super().__init__()
        self.save_hyperparameters("encoder_model_name", "dropout_rate", "k", "max_batches_train")

        self.encoder = AutoModel.from_pretrained(encoder_model_name)

        self.token_importance_linear = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        self.span_dropout = torch.nn.Dropout(dropout_rate)

        self.coarse_bilinear = torch.nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.coarse_dropout = torch.nn.Dropout(dropout_rate)

        self.fine_linear = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size * 3, self.encoder.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.encoder.config.hidden_size, 1)
        )

        self.k = k
        self.max_batches_train = max_batches_train

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CorefModel")
        parser.add_argument("--encoder_model_name", default="sberbank-ai/ruRoberta-large")
        parser.add_argument("--dropout_rate", type=float, default=0.3)
        parser.add_argument("--k", type=int, default=50)
        parser.add_argument("--max_batches_train", type=int)
        return parent_parser

    def forward(self, doc: Doc) -> Tuple[torch.Tensor, torch.Tensor]:

        # Encoding spans ################

        # The input batches of the document are processed independently by a BERT-like model,
        # then the last hidden states of all the batch outputs are taken, exluding special tokens (cls, sep and pad)
        # This gives us embs - [n_tokens, emb] matrix with token embeddings
        input_ids, attention_mask = doc.encoding.input_ids.to(self.device), doc.encoding.attention_mask.to(self.device)
        if not self.training or self.max_batches_train is None or len(input_ids) <= self.max_batches_train:
            embs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            embs = torch.zeros((*input_ids.shape, self.encoder.config.hidden_size), device=self.device)
            selected_batches = torch.rand(len(input_ids)).topk(self.max_batches_train).indices
            embs[selected_batches] = self.encoder(input_ids=input_ids[selected_batches],
                                                  attention_mask=attention_mask[selected_batches]).last_hidden_state
        embs = embs[doc.flattening_mask]                                                    # [n_tokens, emb]

        # We transform a matrix of span starts and ends into an [n_spans, n_tokens] boolean mask,
        # which for i-th row will have 1 at positions of tokens that are part of the i-th span
        spans = torch.tensor(list(doc.all_spans), dtype=torch.long, device=self.device)     # [n_spans, 2]
        indices = torch.arange(0, len(embs), device=self.device).unsqueeze(0).expand(len(spans), len(embs))
        span_mask = (indices >= spans[:, 0].unsqueeze(1)) * (indices < spans[:, 1].unsqueeze(1))
        span_mask = torch.log(span_mask.to(torch.float))

        # Each token representation is passed through trainable token importance linear layer
        # The obtained scores are then softmaxed for each span
        # This way, if a span consists of one token, this token's embeddings will have 1.0 of the weight
        # While, for example, for a two-token span with scores of [1.2, 2.6] the weights will be [0.2, 0.8]
        token_scores = self.token_importance_linear(embs).squeeze(1)                        # [n_tokens]
        token_scores = token_scores.unsqueeze(0).expand(len(spans), len(embs))              # [n_spans, n_tokens]
        token_scores = torch.softmax(token_scores + span_mask, dim=1)

        # Span representations are obtained as weighted sums of the token representations of the span
        embs = token_scores.mm(embs)                                                        # [n_spans, emb]
        embs = self.span_dropout(embs)

        # Coarse span pair scoring ######

        # We set to -inf the scores of all span pairs (i, j) where i <= j (we only predict right to left links)
        pair_mask = torch.arange(0, len(embs), device=self.device)
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))                              # [n_spans, n_spans]

        # Coarse coreference scores are obtained as S ⋅ W ⋅ S.T,
        #   where S is the matrix of span representations
        #   and W is a matrix of trainable weights
        # The pair mask is added to the scores to mask the links between undesired positions
        coarse_scores = self.coarse_dropout(self.coarse_bilinear(embs)).mm(embs.T)
        coarse_scores = pair_mask + coarse_scores                                           # [n_spans, n_spans]

        # Top scoring antecedents are taken further for expensive fine span pair scoring
        top_scores, top_indices = torch.topk(coarse_scores,                                 # [n_spans, n_ants]
                                             k=min(self.k, len(coarse_scores)),
                                             dim=1, sorted=False)

        # Fine span pair scoring ########

        # Fine scores are obtained by first building the pair matrix: a concatenation of the following:
        #   a_spans: embeddings of all spans in the documents
        #   b_spans: embeddings of top-k antecedents for each span of the document
        #   similarity: element-wise product of a_spans and b_spans
        a_spans = embs.unsqueeze(1).expand(embs.shape[0], top_scores.shape[1], embs.shape[1])
        b_spans = embs[top_indices]
        similarity = a_spans * b_spans
        pair_matrix = torch.cat((a_spans, b_spans, similarity), dim=2)                  # [n_spans, n_ants, pair_emb]

        # The resulting pair matrix is passed through dense layers
        fine_scores = self.fine_linear(pair_matrix).squeeze(2)                          # [n_spans, n_ants]

        # Fine scores and coarse scores are added together (important for training)
        return fine_scores + top_scores, top_indices

    @staticmethod
    def lea(a_clusters: List[List[Tuple[int, int]]],
            b_clusters: List[List[Tuple[int, int]]]) -> LEAResult:
        recall, r_weight = CorefModel._lea(a_clusters, b_clusters)
        precision, p_weight = CorefModel._lea(b_clusters, a_clusters)
        return LEAResult(precision, p_weight, recall, r_weight)

    @staticmethod
    def _lea(key: List[List[Tuple[int, int]]],
             response: List[List[Tuple[int, int]]]) -> Tuple[float, float]:
        """ See aclweb.org/anthology/P16-1060.pdf. """
        response_clusters = [set(cluster) for cluster in response]
        response_map = {mention: cluster
                        for cluster in response_clusters
                        for mention in cluster}
        importances = []
        resolutions = []
        for entity in key:
            size = len(entity)
            if size == 1:  # entities of size 1 are not annotated
                continue
            importances.append(size)
            correct_links = 0
            for i in range(size):
                for j in range(i + 1, size):
                    correct_links += int(entity[i] in response_map.get(entity[j], {}))
            resolutions.append(correct_links / (size * (size - 1) / 2))
        res = sum(imp * res for imp, res in zip(importances, resolutions))
        weight = sum(importances)
        return res, weight

    def loss(self, doc: Doc, top_scores: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
        span2entity = {span: i for i, entity in enumerate(doc.entities, start=1) for span in entity}
        entity_ids = torch.tensor([span2entity.get(span, 0) for span in doc.all_spans], device=self.device)

        valid_pair_map = (top_scores > float('-inf'))
        y = entity_ids[top_indices] * valid_pair_map
        y[y == 0] = -1
        y = (y == entity_ids.unsqueeze(1))
        y = torch.cat((y, torch.full((len(y), 1), False, device=y.device)), dim=1)
        y[y.sum(dim=1) == 0, -1] = True
        y = y.to(torch.float)  # [n_spans, k + 1]

        top_scores = torch.cat((top_scores, torch.zeros((len(top_scores), 1), device=top_scores.device)), dim=1)

        gold = torch.logsumexp(top_scores + torch.log(y.to(torch.float)), dim=1)
        pred = torch.logsumexp(top_scores, dim=1)
        return (pred - gold).mean()

    def predict(self, doc: Doc) -> List[List[Tuple[int, int]]]:
        top_scores, top_indices = self(doc)
        positive_scores = (top_scores > 0).detach().cpu().numpy()
        best_ants = top_scores.argmax(dim=1).cpu().numpy()
        indices_map = top_indices.cpu().numpy()

        spans = list(doc.all_spans)
        links = []
        for a_idx, b_ant_idx in enumerate(best_ants):
            if positive_scores[a_idx, b_ant_idx]:
                b_idx = indices_map[a_idx, b_ant_idx]
                links.append((spans[a_idx], spans[b_idx]))

        span2entity = {}

        def get_entity(span: Tuple[int, int]) -> List[Tuple[int, int]]:
            if span not in span2entity:
                span2entity[span] = [span]
            return span2entity[span]

        for source, target in links:
            source_entity, target_entity = get_entity(source), get_entity(target)
            if source_entity is not target_entity:
                source_entity.extend(target_entity)
                for span in target_entity:
                    span2entity[span] = source_entity

        ids = set()
        entities = []
        for entity in span2entity.values():
            if id(entity) not in ids:
                ids.add(id(entity))
                entities.append(entity)

        return sorted(sorted(entity) for entity in entities)

    def run(self, doc: Doc) -> torch.Tensor:
        top_scores, top_indices = self(doc)
        return self.loss(doc, top_scores, top_indices)

    def training_step(self, batch: List[Doc], batch_idx: int):
        if len(batch) > 1:
            loss = torch.cat([self.run(doc) for doc in batch]).sum()
        else:
            loss = self.run(batch[0])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: List[Doc], batch_idx: int):
        scores = []
        for doc in batch:
            scores.append(self.lea(self.predict(doc), doc.entities))
        return scores

    def validation_epoch_end(self, outputs: List[List[LEAResult]]):  # type: ignore[override]
        precision = 0.0
        precision_weight = 0.0
        recall = 0.0
        recall_weight = 0.0
        for output in outputs:
            for result in output:
                precision += result.precision
                precision_weight += result.precision_weight
                recall += result.recall
                recall_weight += result.recall_weight

        total_precision = precision / (precision_weight + EPS)
        total_recall = recall / (recall_weight + EPS)
        f1 = (total_precision * total_recall) / (total_precision + total_recall + EPS) * 2
        self.log("val_lea", f1)

    def configure_optimizers(self):
        coref_parameters = []
        for submodule in self.children():
            if submodule is not self.encoder:
                coref_parameters.extend(submodule.parameters())
        return torch.optim.Adam([
            {"params": self.encoder.parameters(), "lr": 1e-5},
            {"params": coref_parameters, "lr": 3e-4}
        ])
