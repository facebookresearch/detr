# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch

from models.matcher import HungarianMatcher
from models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from models.backbone import Backbone, Joiner, BackboneBase
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from hubconf import detr_resnet50, detr_resnet50_panoptic


class Tester(unittest.TestCase):

    def test_box_cxcywh_to_xyxy(self):
        t = torch.rand(10, 4)
        r = box_ops.box_xyxy_to_cxcywh(box_ops.box_cxcywh_to_xyxy(t))
        self.assertLess((t - r).abs().max(), 1e-5)

    @staticmethod
    def indices_torch2python(indices):
        return [(i.tolist(), j.tolist()) for i, j in indices]

    def test_hungarian(self):
        n_queries, n_targets, n_classes = 100, 15, 91
        logits = torch.rand(1, n_queries, n_classes + 1)
        boxes = torch.rand(1, n_queries, 4)
        tgt_labels = torch.randint(high=n_classes, size=(n_targets,))
        tgt_boxes = torch.rand(n_targets, 4)
        matcher = HungarianMatcher()
        targets = [{'labels': tgt_labels, 'boxes': tgt_boxes}]
        indices_single = matcher({'pred_logits': logits, 'pred_boxes': boxes}, targets)
        indices_batched = matcher({'pred_logits': logits.repeat(2, 1, 1),
                                   'pred_boxes': boxes.repeat(2, 1, 1)}, targets * 2)
        self.assertEqual(len(indices_single[0][0]), n_targets)
        self.assertEqual(len(indices_single[0][1]), n_targets)
        self.assertEqual(self.indices_torch2python(indices_single),
                         self.indices_torch2python([indices_batched[0]]))
        self.assertEqual(self.indices_torch2python(indices_single),
                         self.indices_torch2python([indices_batched[1]]))

        # test with empty targets
        tgt_labels_empty = torch.randint(high=n_classes, size=(0,))
        tgt_boxes_empty = torch.rand(0, 4)
        targets_empty = [{'labels': tgt_labels_empty, 'boxes': tgt_boxes_empty}]
        indices = matcher({'pred_logits': logits.repeat(2, 1, 1),
                           'pred_boxes': boxes.repeat(2, 1, 1)}, targets + targets_empty)
        self.assertEqual(len(indices[1][0]), 0)
        indices = matcher({'pred_logits': logits.repeat(2, 1, 1),
                           'pred_boxes': boxes.repeat(2, 1, 1)}, targets_empty * 2)
        self.assertEqual(len(indices[0][0]), 0)

    def test_position_encoding_script(self):
        m1, m2 = PositionEmbeddingSine(), PositionEmbeddingLearned()
        mm1, mm2 = torch.jit.script(m1), torch.jit.script(m2)  # noqa

    def test_backbone_script(self):
        backbone = Backbone('resnet50', True, False, False)
        torch.jit.script(backbone)  # noqa

    def test_model_script_detection(self):
        model = detr_resnet50(pretrained=False).eval()
        scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))

    def test_model_script_panoptic(self):
        model = detr_resnet50_panoptic(pretrained=False).eval()
        scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))
        self.assertTrue(out["pred_masks"].equal(out_script["pred_masks"]))


if __name__ == '__main__':
    unittest.main()
