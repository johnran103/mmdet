# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from new_api import imshow_det_bboxes

import numpy as np

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.05, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    labels = []
    for i in range(len(result)):
        sz = [i] * len(result[i])
        labels.extend(sz)
    print(len(labels))
    # show the results
    #show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    # imshow_det_bboxes(args.img, np.concatenate(result), labels=labels,
    # class_names=('pedestrian', 'people', 'bicycle', 'car', 'van','truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
    # ,out_file='./demo.jpg')

    imshow_det_bboxes(args.img, np.concatenate(result), labels=labels,class_names=('car'),out_file='./demo/' + args.img, score_thr=0.2)



async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    # show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
