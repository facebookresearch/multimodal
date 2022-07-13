# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluator for Flickr30k """
import xml.etree.ElementTree as Et
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import utils.dist as dist
from prettytable import PrettyTable

from torch import Tensor
from torchvision.ops.boxes import box_iou


def get_sentence_data(filename) -> List[Dict[str, Any]]:
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      filename - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(filename, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(
            first_word, phrases, phrase_id, phrase_type
        ):
            sentence_data["phrases"].append(
                {
                    "first_word_index": index,
                    "phrase": phrase,
                    "phrase_id": p_id,
                    "phrase_type": p_type,
                }
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(
    filename,
) -> Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]]:
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      filename - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
          height - int representing the height of the image
          width - int representing the width of the image
          depth - int representing the depth of the image
    """
    tree = Et.parse(filename)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info: Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]] = {}
    all_boxes: Dict[str, List[List[int]]] = {}
    all_noboxes: List[str] = []
    all_scenes: List[str] = []
    for size_element in size_container:
        assert size_element.text
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            assert box_id
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in all_boxes:
                    all_boxes[box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text)
                ymin = int(box_container[0].findall("ymin")[0].text)
                xmax = int(box_container[0].findall("xmax")[0].text)
                ymax = int(box_container[0].findall("ymax")[0].text)
                all_boxes[box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    all_noboxes.append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    all_scenes.append(box_id)
    anno_info["boxes"] = all_boxes
    anno_info["nobox"] = all_noboxes
    anno_info["scene"] = all_scenes

    return anno_info


def _merge_boxes(boxes: List[List[int]]) -> List[List[int]]:
    """
    Return the boxes corresponding to the smallest enclosing box containing all the provided boxes
    The boxes are expected in [x1, y1, x2, y2] format
    """
    if len(boxes) == 1:
        return boxes

    np_boxes = np.asarray(boxes)

    return [
        [
            np_boxes[:, 0].min(),
            np_boxes[:, 1].min(),
            np_boxes[:, 2].max(),
            np_boxes[:, 3].max(),
        ]
    ]


class RecallTracker:
    """Utility class to track recall@k for various k, split by categories"""

    def __init__(self, topk: Sequence[int]):
        """
        Parameters:
           - topk : tuple of ints corresponding to the recalls being tracked (eg, recall@1, recall@10, ...)
        """

        self.total_byk_bycat: Dict[int, Dict[str, int]] = {
            k: defaultdict(int) for k in topk
        }
        self.positives_byk_bycat: Dict[int, Dict[str, int]] = {
            k: defaultdict(int) for k in topk
        }

    def add_positive(self, k: int, category: str):
        """Log a positive hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1
        self.positives_byk_bycat[k][category] += 1

    def add_negative(self, k: int, category: str):
        """Log a negative hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1

    def report(self) -> Dict[int, Dict[str, float]]:
        """Return a condensed report of the results as a dict of dict.
        report[k][cat] is the recall@k for the given category
        """
        report: Dict[int, Dict[str, float]] = {}
        for k in self.total_byk_bycat:
            assert k in self.positives_byk_bycat
            report[k] = {
                cat: self.positives_byk_bycat[k][cat] / self.total_byk_bycat[k][cat]
                for cat in self.total_byk_bycat[k]
            }
        return report


class Flickr30kEntitiesRecallEvaluator:
    def __init__(
        self,
        flickr_path: str,
        subset: str = "test",
        topk: Sequence[int] = (1, 5, 10, -1),
        iou_thresh: float = 0.5,
        merge_boxes: bool = False,
    ):

        assert subset in ["train", "test", "val"], f"Wrong flickr subset {subset}"

        self.topk = topk
        self.iou_thresh = iou_thresh

        flickr_path = Path(flickr_path)

        # Load the image ids corresponding to the current subset
        with open(flickr_path / f"{subset}.txt") as file_d:
            self.img_ids = [line.strip() for line in file_d]

        # Read the box annotations for all the images
        self.imgid2boxes: Dict[str, Dict[str, List[List[int]]]] = {}

        for img_id in self.img_ids:
            anno_info = get_annotations(flickr_path / "Annotations" / f"{img_id}.xml")[
                "boxes"
            ]
            if merge_boxes:
                merged = {}
                for phrase_id, boxes in anno_info.items():
                    merged[phrase_id] = _merge_boxes(boxes)
                anno_info = merged
            self.imgid2boxes[img_id] = anno_info

        # Read the sentences annotations
        self.imgid2sentences: Dict[str, List[List[Optional[Dict]]]] = {}

        self.all_ids: List[str] = []
        tot_phrases = 0
        for img_id in self.img_ids:
            sentence_info = get_sentence_data(
                flickr_path / "Sentences" / f"{img_id}.txt"
            )
            self.imgid2sentences[img_id] = [None for _ in range(len(sentence_info))]

            # Some phrases don't have boxes, we filter them.
            for sent_id, sentence in enumerate(sentence_info):
                phrases = [
                    phrase
                    for phrase in sentence["phrases"]
                    if phrase["phrase_id"] in self.imgid2boxes[img_id]
                ]
                if len(phrases) > 0:
                    self.imgid2sentences[img_id][sent_id] = phrases
                tot_phrases += len(phrases)

            self.all_ids += [
                f"{img_id}_{k}"
                for k in range(len(sentence_info))
                if self.imgid2sentences[img_id][k] is not None
            ]

    def evaluate(self, predictions: List[Dict]):
        evaluated_ids = set()

        recall_tracker = RecallTracker(self.topk)

        for pred in predictions:
            cur_id = f"{pred['image_id']}_{pred['sentence_id']}"
            if cur_id in evaluated_ids:
                print(
                    "Warning, multiple predictions found for sentence"
                    f"{pred['sentence_id']} in image {pred['image_id']}"
                )
                continue

            # Skip the sentences with no valid phrase
            if cur_id not in self.all_ids:
                if len(pred["boxes"]) != 0:
                    print(
                        f"Warning, in image {pred['image_id']} we were not expecting predictions "
                        f"for sentence {pred['sentence_id']}. Ignoring them."
                    )
                continue

            evaluated_ids.add(cur_id)

            pred_boxes = pred["boxes"]
            if str(pred["image_id"]) not in self.imgid2sentences:
                raise RuntimeError(f"Unknown image id {pred['image_id']}")
            if (
                not 0
                <= int(pred["sentence_id"])
                < len(self.imgid2sentences[str(pred["image_id"])])
            ):
                raise RuntimeError(
                    f"Unknown sentence id {pred['sentence_id']}"
                    f" in image {pred['image_id']}"
                )

            phrases = self.imgid2sentences[str(pred["image_id"])][
                int(pred["sentence_id"])
            ]
            if len(pred_boxes) != len(phrases):
                raise RuntimeError(
                    f"Error, got {len(pred_boxes)} predictions, expected {len(phrases)} "
                    f"for sentence {pred['sentence_id']} in image {pred['image_id']}"
                )

            for cur_boxes, phrase in zip(pred_boxes, phrases):
                target_boxes = self.imgid2boxes[str(pred["image_id"])][
                    phrase["phrase_id"]
                ]
                ious = box_iou(Tensor(cur_boxes), Tensor(target_boxes))
                for k in self.topk:
                    maxi = 0
                    if k == -1:
                        maxi = ious.max()
                    else:
                        assert k > 0
                        maxi = ious[:k].max()
                    if maxi >= self.iou_thresh:
                        recall_tracker.add_positive(k, "all")
                        for phrase_type in phrase["phrase_type"]:
                            recall_tracker.add_positive(k, phrase_type)
                    else:
                        recall_tracker.add_negative(k, "all")
                        for phrase_type in phrase["phrase_type"]:
                            recall_tracker.add_negative(k, phrase_type)

        if len(evaluated_ids) != len(self.all_ids):
            print(
                "ERROR, the number of evaluated sentence doesn't match. Missing predictions:"
            )
            un_processed = set(self.all_ids) - evaluated_ids
            for missing in un_processed:
                img_id, sent_id = missing.split("_")
                print(f"\t sentence {sent_id} in image {img_id}")
            raise RuntimeError("Missing predictions")

        return recall_tracker.report()


class FlickrEvaluator(object):
    def __init__(
        self,
        flickr_path,
        subset,
        top_k=(1, 5, 10, -1),
        iou_thresh=0.5,
        merge_boxes=False,
    ):
        assert isinstance(top_k, (list, tuple))

        self.evaluator = Flickr30kEntitiesRecallEvaluator(
            flickr_path,
            subset=subset,
            topk=top_k,
            iou_thresh=iou_thresh,
            merge_boxes=merge_boxes,
        )
        self.predictions = []
        self.results = None

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions += predictions

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        self.predictions = sum(all_predictions, [])

    def summarize(self):
        if dist.is_main_process():
            self.results = self.evaluator.evaluate(self.predictions)
            table = PrettyTable()
            all_cat = sorted(list(self.results.values())[0].keys())
            table.field_names = ["Recall@k"] + all_cat

            score = {}
            for k, v in self.results.items():
                cur_results = [v[cat] for cat in all_cat]
                header = "Upper_bound" if k == -1 else f"Recall@{k}"

                for cat in all_cat:
                    score[f"{header}_{cat}"] = v[cat]
                table.add_row([header] + cur_results)

            print(table)

            return score

        return None, None
