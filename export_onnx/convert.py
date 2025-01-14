import cv2
import numpy as np
import os
import time
import argparse
from onnxmodel import ONNXModel
import torch
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import load_network
from packnet_sfm.models.SfmModel import SfmModel
from utils.file import WalkImage, MkdirSimple
from export_onnx.onnx_test import test_dir

W = 640
H = 384


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device to run the model on.")
    parser.add_argument("--width", type=int, default=W, help="Width of the input image.")
    parser.add_argument("--height", type=int, default=H, help="Height of the input image.")
    parser.add_argument("--test", action="store_true", help="test model")
    return parser.parse_args()


def load_model(model_path, device):
    config, state_dict = parse_test_file(model_path)
    model = SfmModel()

    # Add depth network if required
    if 'depth_net' in model.network_requirements:
        version = config.model.depth_net.version
        depth_net_module = __import__('packnet_sfm.networks.depth', fromlist=[config.model.depth_net['name']])
        depth_net_class = getattr(getattr(depth_net_module, config.model.depth_net['name']), config.model.depth_net['name'])
        depth_net = depth_net_class(version)
        depth_net = load_network(depth_net, model_path, 'depth_net')

    # Add pose network if required
    if 'pose_net' in model.network_requirements:
        version = config.model.depth_net.version
        pose_net_module = __import__('packnet_sfm.networks.pose', fromlist=[config.model.pose_net['name']])
        pose_net_class = getattr(getattr(pose_net_module, config.model.pose_net['name']), config.model.pose_net['name'])
        pose_net = pose_net_class(version)
        pose_net = load_network(pose_net, model_path, 'pose_net')
        model.add_pose_net(pose_net)

    # model = load_network(model, config.checkpoint_path, 'model')
    # model.load_state_dict(state_dict)
    # model = model.to(device).eval()

    depth_net = depth_net.to(device).eval()
    pose_net = pose_net.to(device).eval()
    return depth_net, pose_net


def export_to_onnx(model_path, onnx_file, width=W, height=H, device="cuda"):
    model, pose = load_model(model_path, device)

    # Create dummy input for the model
    dummy_input = torch.randn(1, 3, height, width).to(device)  # Adjust the size as needed
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device {param.device}")
        break
    print(f"dummy_input is on device {dummy_input.device}")
    # Export the model
    torch.onnx.export(model, dummy_input, onnx_file,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True)

    print(f"Model exported to {onnx_file}")
    ONNXModel(onnx_file)


def main():
    args = parse_args()
    args = parse_args()
    model_name = os.path.splitext(os.path.basename(args.checkpoint))[0].replace(" ", "_")
    output = os.path.join(args.output, model_name, f'{args.width}_{args.height}')
    onnx_file = os.path.join(output, f'DepthAnythingV2_{args.width}_{args.height}_{model_name}_12.onnx')
    MkdirSimple(output)

    export_to_onnx(args.checkpoint, onnx_file, args.width, args.height, args.device)  # Replace 'vitl' with the desired encoder

    print("export onnx to {}".format(onnx_file))
    if args.test:
        test_dir(onnx_file, [args.image, ], output)



if __name__ == "__main__":
    main()
