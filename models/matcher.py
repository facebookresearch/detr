# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # In the comments below:
        # - `bs` is the batch size, i.e. outputs["pred_logits"].shape[0];
        # - `mo` is the maximum number of objects over all the targets,
        # i.e. `max((len(v["labels"]) for v in targets))`;
        # - `q` is the number of queries, i.e. outputs["pred_logits"].shape[1];
        # - `cl` is the number of classes including no-object,
        # i.e. outputs["pred_logits"].shape[2] or self.num_classes + 1.
        if len(targets) == 1:
            # This branch is just an optimization, not needed for correctness.
            tgt_ids = targets[0]["labels"].unsqueeze(dim=0)
            tgt_bbox = targets[0]["boxes"].unsqueeze(dim=0)
        else:
            tgt_ids = pad_sequence(
                [target["labels"] for target in targets],
                batch_first=True,
                padding_value=0
            )  # (bs, mo)
            tgt_bbox = pad_sequence(
                [target["boxes"] for target in targets],
                batch_first=True,
                padding_value=0
            )  # (bs, mo, 4)

        out_bbox = outputs["pred_boxes"]  # (bs, q, 4)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (bs, q, mo)

        # Compute the giou cost betwen boxes
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        giou = generalized_box_iou(
            out_bbox_xyxy, tgt_bbox_xyxy)  # (bs, q, mo)

        # Compute the classification cost. Contrary to the loss, we don't use
        # the Negative Log Likelihood, but approximate it
        # in `1 - proba[target class]`. The 1 is a constant that does not
        # change the matching, it can be ommitted.
        out_prob = outputs["pred_logits"].softmax(-1)  # (bs, q, c)
        prob_class = torch.gather(
            out_prob,
            dim=2,
            index=tgt_ids.unsqueeze(dim=1).expand(-1, out_prob.shape[1], -1)
        )  # (bs, q, mo)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox - self.cost_giou * giou - self.cost_class * prob_class
        c = C.cpu()

        indices = [
            linear_sum_assignment(c[i, :, :len(v["labels"])])
            for i, v in enumerate(targets)
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
