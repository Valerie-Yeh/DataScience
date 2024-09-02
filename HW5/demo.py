import argparse
from pathlib import Path

from PIL import Image
from psgan import Inference
from fire import Fire
import numpy as np

import faceutils as futils
from psgan import PostProcess
from setup import setup_config, setup_argparser
import os
import shutil

parser = setup_argparser()
parser.add_argument("--device", default="cuda", help="device used for inference")
parser.add_argument("--model_path", default="G.pth", help="model for loading")
parser.add_argument("--image_path", default="../mtdataset/images/", help="image folder")
parser.add_argument("--output_path", default="output", help="output path for storing")
parser.add_argument("--makeup", default="../makeup_test.txt", help="makeup text, order for generation")
parser.add_argument("--non_makeup", default="../nomakeup_test.txt", help="non_makeup text, order for generation")

args = parser.parse_args()


def main(save_path, filename, filename_n):
    
    config = setup_config(args)

    # Using the second cpu
    inference = Inference(config, args.device, args.model_path)
    postprocess = PostProcess(config)
    source = Image.open(args.image_path + filename_n).convert("RGB")
    reference = Image.open(args.image_path + filename).convert("RGB")
  
    # Transfer the psgan from reference to source.
    image, face = inference.transfer(source, reference, with_face=True)
    if face == None:
        print(f"Can't identify face, skip {save_path}")
    else:
        source_crop = source.crop((face.left(), face.top(), face.right(), face.bottom()))
        image = postprocess(source_crop, image)
        image = image.resize((128,128))
        image.save(save_path)


if __name__ == '__main__':
    out_path = args.output_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file = open(args.makeup, 'r')
    file_n = open(args.non_makeup, 'r')
    makeups = file.readlines()
    non_makeups = file_n.readlines()
    
    for i in range(len(makeups)):
        filename = makeups[i].split()[0]
        filename_n = non_makeups[i].split()[0]
        main(out_path + '/' + f'pred_{i}.png', filename, filename_n)