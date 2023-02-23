import sys
sys.path.append('core')
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from LLcvnet import LLcvNetEighth
from utils.utils import InputPadder
from utils import frame_utils

DEVICE = 'cuda'

def load_image(imfile):

    img = cv2.imread(imfile)
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output', default='',help="flow file")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--num_k', type=int, default=32,
                        help='number of hypotheses to compute for knn Faiss')
    args = parser.parse_args()

    model = torch.nn.DataParallel(LLcvNetEighth(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join())
        images = sorted(images)
        i = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            print(imfile1,imfile2)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            frame_utils.writeFlow(os.path.join(args.output,'{}_flow.flo'.format(str(i))), flow)
            i+=2

