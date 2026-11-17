from __future__ import absolute_import, division, print_function
import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms, datasets
import networks
import time
from utils import select_device

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def parse_args():
    parser = argparse.ArgumentParser(description='Simple testing funtion for R-MSFM models.')
    parser.add_argument('--image_path', type=str,help='path to a test image or folder of images', required=True)
    parser.add_argument('--ext', type=str,help='image extension to search for in folder', default="jpeg")
    parser.add_argument('--model_path', type=str,help='path to a models.pth', default="./3M")
    parser.add_argument('--iters', type=int,help='iterative update', default=6)
    parser.add_argument("--no_cuda",help='if set, disables CUDA',action='store_true')
    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="efficientnet-b1",
                        choices=[
                            "efficientnet-b0",
                            "efficientnet-b1"])

    return parser.parse_args()

def test_simple(args):
    device = select_device(args.no_cuda)

    model_path = args.model_path
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']
    print("   Loading pretrained encoder")
    encoder = networks.EfficientEncoder(valid_models=args.model, pretrained=False,
                                       height=feed_height, width=feed_width)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")


    depth_decoder = networks.DepthDecoder.from_checkpoint(
        encoder.num_ch_enc, args.iters,
        torch.load(depth_decoder_path, map_location="cpu", weights_only=True))
    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        paths = [args.image_path]
        output_directory = os.path.join(os.path.dirname(args.image_path), 'output')
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = os.path.join(args.image_path, 'output')
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    os.makedirs(output_directory, exist_ok=True)

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        min_infer_time = 10
        n = 0
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.Resampling.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.time()
            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.time()
            infer_time = end-start
            #print('infer_time:', infer_time)

            if infer_time < min_infer_time:
                min_infer_time = infer_time

            if idx >= 299:
                n += infer_time

            disp = outputs[("disp_up", args.iters -1)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100) 
            #np.save(name_dest_npy, scaled_disp.cpu().numpy())
            
            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            im = pil.fromarray(colormapped_im)#转为PIL图像

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(idx + 1, len(paths), name_dest_im))

    n = n/398
    print('averged infer_time:', n)
    print('min_infer_time:', min_infer_time)
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)

'''
python test_simple.py --image_path='/media/penglong/致钛2T/PL/input_c' --model_path='weights/FR-LK0' --iters=2 --ext png

python test_simple.py --image_path='/media/penglong/致钛2T/PL/input' --model_path='weights/1024_b0_4' --iters=4 --model efficientnet-b0

python test_simple.py --image_path='/media/penglong/致钛2T/PL/input_k1' --model_path='weights/FR-LM-GC' --iters=2 

'''
