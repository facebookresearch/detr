import torch
import argparse
from pathlib import Path
from models.detr import DETR
from models import build_model
from predictor import get_args_parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 800, 800, device='cuda')

    device = torch.device(args.device)
    model, _, _ = build_model(args)
    # print(model)
    model.to(device)


    torch.onnx.export(model, dummy_input, "test.onnx", verbose=True, opset_version=11)



# import onnx
#
#
# if __name__ == "__main__":
#     # Load the ONNX model
#     model = onnx.load("test.onnx")
#
#     # Check that the IR is well formed
#     onnx.checker.check_model(model)
#
#     # Print a human readable representation of the graph
#     onnx.helper.printable_graph(model.graph)