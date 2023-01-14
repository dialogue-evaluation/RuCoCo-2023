from collections import defaultdict
import json
import os
import sys
from typing import *


Span = Tuple[int, int]

EPS = 1e-7


class DocumentPair(NamedTuple):
    filename: str
    dir_a: str
    dir_b: str


class ScoringException(Exception):
    pass


def agreement(pairs: Iterable[DocumentPair]) -> Tuple[float, float, float]:
    total_recall, total_r_weight = .0, .0
    total_precision, total_p_weight = .0, .0
    for pair in sorted(pairs):
        a = read_markup_dict(os.path.join(pair.dir_a, pair.filename))
        b = read_markup_dict(os.path.join(pair.dir_b, pair.filename))
        if a["text"] != b["text"]:
            raise ScoringException(f"mismatching texts for documents: {pair.filename} in {pair.dir_a} and {pair.dir_b}")

        a_clusters = [(spans, get_children(a, i))
                      for i, spans in enumerate(a["entities"])]
        b_clusters = [(spans, get_children(b, i))
                      for i, spans in enumerate(b["entities"])]

        recall, r_weight = _lea_children(a_clusters, b_clusters)
        precision, p_weight = _lea_children(b_clusters, a_clusters)

        total_recall += recall
        total_r_weight += r_weight
        total_precision += precision
        total_p_weight += p_weight

    recall = total_recall / (total_r_weight + EPS)
    precision = total_precision / (total_p_weight + EPS)
    return f1(recall, precision), precision, recall


def f1(precision: float, recall: float, eps: float = 1e-7) -> float:
    return (precision * recall) / (precision + recall + eps) * 2


def get_children(data: dict, idx: int) -> List[Span]:
    """ Returns a list of all the immediate AND most distant children """
    children = set()
    for child_idx in data["includes"][idx]:
        children.update(data["entities"][child_idx])

    visited = set()
    stack = list(data["includes"][idx])
    while stack:
        child_idx = stack.pop()
        visited.add(child_idx)
        if not data["includes"][child_idx]:
            children.update(data["entities"][child_idx])
        else:
            for grandchild_idx in data["includes"][child_idx]:
                if grandchild_idx not in visited:
                    stack.append(grandchild_idx)

    return sorted(children)


def get_pairs_from_dir(path: str) -> List[DocumentPair]:
    entries = filter(lambda entry: entry.name.endswith(".json"),
                     recursive_scandir(path))
    name2paths = defaultdict(list)
    for entry in entries:
        name2paths[entry.name].append(entry.path)

    pairs = []
    for name, paths in name2paths.items():
        if len(paths) == 1:
            raise ScoringException(f"No matching document for {paths[0]}")
        elif len(paths) > 2:
            raise ScoringException(f"Too many matching documents: {', '.join(paths)}")
        else:
            pairs.append(
                DocumentPair(name, *(os.path.dirname(path) for path in paths))
            )
    return pairs


def get_pairs_from_two_dirs(a: str,
                            b: str) -> List[DocumentPair]:
    a_files = set(get_relative_paths(a))
    b_files = set(get_relative_paths(b))
    common_files = a_files & b_files

    for file in a_files - common_files:
        raise ScoringException(f"No matching document for {os.path.join(a, file)}")
    for file in b_files - common_files:
        raise ScoringException(f"No matching document for {os.path.join(b, file)}")
    return [DocumentPair(filename, a, b) for filename in common_files]


def get_relative_paths(path: str) -> Iterator[str]:
    return map(lambda entry: os.path.relpath(entry.path, path),
               filter(lambda entry: entry.name.endswith(".json"),
                      recursive_scandir(path)))


def read_markup_dict(path: str) -> dict:
    with open(path, mode="r", encoding="utf8") as f:
        markup_dict = json.load(f)
    markup_dict["entities"] = [[tuple(span) for span in entity]
                               for entity in markup_dict["entities"]]
    return markup_dict


def recursive_scandir(path: str) -> Iterator[os.DirEntry]:
    for entry in os.scandir(path):
        if entry.is_dir():
            yield from recursive_scandir(entry.path)
        else:
            yield entry


def _lea_children(key: List[Tuple[List[Span], List[Span]]],
                  response: List[Tuple[List[Span], List[Span]]]
                  ) -> Tuple[float, float]:
        response_clusters = [set(cluster) for cluster, _ in response]
        response_map = {mention: cluster
                        for cluster in response_clusters
                        for mention in cluster}
        response_children_map = defaultdict(set)
        for cluster, children in response:
            for mention in children:
                response_children_map[mention].update(cluster)

        importances = []
        resolutions = []
        for entity, children in key:
            size = len(entity)
            if size > 1:  # entities of size 1 are not annotated
                importances.append(size)
                correct_links = 0
                for i in range(size):
                    for j in range(i + 1, size):
                        correct_links += int(entity[i]
                                            in response_map.get(entity[j], {}))
                resolutions.append(correct_links / (size * (size - 1) / 2))

            if not children:
                continue
            importances.append(len(children))
            correct_links = 0
            for mention in entity:
                for child in children:
                    correct_links += int(mention in response_children_map.get(child, {}))
            resolutions.append(correct_links / (size * len(children)))

        res = sum(imp * res for imp, res in zip(importances, resolutions))
        weight = sum(importances)
        return res, weight


if __name__ == "__main__":
    _, input_dir, output_dir = sys.argv
    ref_dir = os.path.join(input_dir, "ref")
    res_dir = os.path.join(input_dir, "res")
    scores_path = os.path.join(output_dir, "scores.txt")

    pairs = get_pairs_from_two_dirs(ref_dir, res_dir)

    f1_score, precision, recall = agreement(pairs)

    os.makedirs(output_dir, exist_ok=True)
    with open(scores_path, mode="w", encoding="utf8") as f:
        print(f"F1: {f1_score:.3f}", file=f)
        print(f"Precision: {precision:.3f}", file=f)
        print(f"Recall: {recall:.3f}", file=f)
