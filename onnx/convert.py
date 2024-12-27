import cv2
import torch
import numpy as np
import os
import time
import argparse
from tools.file import MkdirSimple
from tools.utils import print_onnx
from packnet_sfm.models.model_wrapper import ModelWrapper
from onnxmodel import ONNXModel
from packnet_sfm.utils.config import parse_test_file
import torch
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import load_network
from packnet_sfm.networks.depth.DepthResNet import DepthResNet
from packnet_sfm.networks.pose import PoseNet
from packnet_sfm.models.SfmModel import SfmModel

W = 640
H = 384

def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="Device to run the model on.")
    parser.add_argument("--width", type=int, default=W, help="Width of the input image.")
    parser.add_argument("--height", type=int, default=H, help="Height of the input image.")
    return parser.parse_args()




def load_model(model_path, device):
    config, state_dict = parse_test_file(model_path)
    model = SfmModel()

    # Add depth network if required
    if 'depth_net' in model.network_requirements:
        depth_net = DepthResNet(**config.depth_net)
        if config.depth_net.checkpoint_path:
            depth_net = load_network(depth_net, config.depth_net.checkpoint_path, 'depth_net')


    # Add pose network if required
    if 'pose_net' in model.network_requirements:
        pose_net = PoseNet(**config.pose_net)
        if config.pose_net.checkpoint_path:
            pose_net = load_network(pose_net, config.pose_net.checkpoint_path, 'pose_net')
        model.add_pose_net(pose_net)

    # If a checkpoint is provided, load pretrained model
    if config.checkpoint_path:
        model = load_network(model, config.checkpoint_path, 'model')

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    return model

def export_to_onnx(model_path, onnx_file, width=W, height=H, device="cuda"):
    model = load_model(model_path, device)

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
    print_onnx(onnx_file)

def test_onnx(img_path, model_file, width=W, height=H, device="cuda"):
    model = ONNXModel(model_file)
    img_org = load_image(img_path)
    img = resize_image(img_org, (height, width))
    img = to_tensor(img).unsqueeze(0).numpy()

    start_time = time.time()
    output = model.forward(img)
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
    dis_array = output[0][0]
    dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
    dis_array = dis_array.astype("uint8")

    depth = cv2.resize(dis_array, (img_org.shape[1], img_org.shape[0]))
    depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, 1), cv2.COLORMAP_PARULA)
    combined_img = np.vstack((img_org, depth))

    return combined_img, depth

def main():
    args = parse_args()
    output_dir = os.path.join(args.output, f"{args.width}_{args.height}")
    onnx_file = os.path.join(output_dir, os.path.splitext(os.path.basename(args.checkpoint))[0] + ".onnx")
    MkdirSimple(onnx_file)
    export_to_onnx(args.checkpoint, onnx_file, args.width, args.height, args.device)
    image, depth = test_onnx(args.image, onnx_file, args.width, args.height, 'cuda')
    cv2.imwrite(os.path.join(output_dir, "test.png"), image)
    cv2.imwrite(os.path.join(output_dir, "depth.png"), depth)

if __name__ == "__main__":
    main()