from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
import time

parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--img_file', help='dataset')
parser.add_argument('--confidence_threshold', default=0.8, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def draw_detections_with_conf(image, dets):
    if dets.shape[0] == 0:
        return image
    for k in range(dets.shape[0]):
        cv2.rectangle(image, (int(dets[k, 0]), int(dets[k, 1])), (int(dets[k, 2]), int(dets[k, 3])), (0, 255, 0), 3)
        #startY = int(dets[k, 1])
        #y = startY - 15 if startY - 15 > 15 else startY + 15
        #cv2.putText(image, "%.1f" % int(dets[k, 4]), (int(dets[k, 0]), y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
    return image


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    resize = 1

    start_time =time.time()
    img_name = args.img_file
    img_orig = cv2.imread(img_name, cv2.IMREAD_COLOR)
    img = np.float32(cv2.imread(img_name, cv2.IMREAD_COLOR))
    im_height, im_width, _ = img.shape
    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    # keep = py_cpu_nms(dets, args.nms_threshold)
    keep = nms(dets, args.nms_threshold, force_cpu=args.cpu)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]

    for k in range(dets.shape[0]):
        xmin = dets[k, 0]
        ymin = dets[k, 1]
        xmax = dets[k, 2]
        ymax = dets[k, 3]
        ymin += 0.2 * (ymax - ymin + 1)
        score = dets[k, 4]
    vis_image = draw_detections_with_conf(img_orig, dets)
    cv2.imwrite('test.png', vis_image)

    end_time = time.time()
    print('Time Taken: ', end_time-start_time, ' seconds')