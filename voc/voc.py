import os
import sys
import pickle
import torch
import tempfile
import logging 
import torch.nn as nn
import numpy as np

from utils.config import Config
from voc.voc_eval import voc_eval
from collections import ChainMap
from utils.dist import gather, synchronize, is_main_process

logger = logging.getLogger(__name__)

def voc_ap_metrics(data_list ,forward_time, nms_time, test_dataloader, distributed = False):
    statistics = torch.cuda.FloatTensor([forward_time, nms_time, len(test_dataloader) - 1])
    if distributed:
        data_list = gather(data_list, dst=0)
        data_list = ChainMap(*data_list)
        torch.distributed.reduce(statistics, dst=0)
    mAP50, mAP5095, summary = evaluate_prediction(data_list, statistics, test_dataloader)
    synchronize()
    logger.info(summary)
    return mAP50, mAP5095

def evaluate_prediction(data_dict, statistics, test_dataloader):
    if not is_main_process():
        return 0, 0, None
    # logger.info("Evaluate in main process...")
    forward_time = statistics[0].item()
    nms_time = statistics[1].item()
    n_samples = statistics[-1].item()
    a_forward_time = 1000 * forward_time / (n_samples * test_dataloader.batch_size)
    a_nms_time = 1000 * nms_time / (n_samples * test_dataloader.batch_size)
    time_info = ", ".join(
        [
            "Average {} time: {:.2f} ms".format(k, v)
            for k, v in zip(
                ["Forward", "NMS", "Inference"],
                [a_forward_time, a_nms_time, (a_forward_time + a_nms_time)],
            )
        ]
    )

    info = time_info + "\n"

    all_boxes = [
        [[] for _ in range(len(test_dataloader.dataset))] for _ in range(len(test_dataloader.dataset.CLASSES_NAME))
    ]
    for img_num in range(len(test_dataloader.dataset)):
        bboxes, cls, scores = data_dict[img_num]
        if bboxes is None:
            for j in range(len(test_dataloader.dataset.CLASSES_NAME)):
                all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
            continue
        for j in range(len(test_dataloader.dataset.CLASSES_NAME)):
            mask_c = cls == j       # np(nums_keep, )
            if sum(mask_c) == 0:
                all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                continue
            c_dets = np.hstack((bboxes, scores[:,None]))
            all_boxes[j][img_num] = c_dets[mask_c]

        sys.stdout.write(
            "im_eval: {:d}/{:d} \r".format(img_num + 1, len(test_dataloader.dataset))
        )
        sys.stdout.flush()

    with tempfile.TemporaryDirectory() as tempdir:
        mAP50, mAP5095 = evaluate_detections(
            all_boxes, test_dataloader, tempdir
        )
        return mAP50, mAP5095, info

def evaluate_detections(all_boxes, test_dataloader, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        write_voc_results_file(all_boxes, test_dataloader)
        IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        mAPs = []
        logging.info("\n")
        for iou in IouTh:
            mAP = do_python_eval(test_dataloader, output_dir, iou)
            mAPs.append(mAP)

        print("\n--------------------------------------------------------------")
        print("Map_5095:", np.mean(mAPs))
        print("Map_50  :", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

def write_voc_results_file(all_boxes, test_dataloader):
    for cls_ind, cls in enumerate(test_dataloader.dataset.CLASSES_NAME):
        cls_ind = cls_ind
        if cls == "__background__":
            continue
        # print("Writing {} VOC results file".format(cls))
        filename = get_voc_results_file_template().format(cls)  #write in /data-nbd/wuxc/111111/voc/result
        with open(filename, "wt") as f:
            for im_ind, index in enumerate(test_dataloader.dataset.img_ids):
                index = index[1]
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    f.write(
                        "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                            index,              # pic name
                            dets[k, -1],        # conf
                            dets[k, 0] + 1,
                            dets[k, 1] + 1,
                            dets[k, 2] + 1,
                            dets[k, 3] + 1,
                        )
                    )

def get_voc_results_file_template():
    filename = "det_test" + "_{:s}.txt"
    filedir = os.path.join(Config.testresults_root, "results")
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

def do_python_eval(test_dataloader, output_dir="output", iou=0.5):
    annopath = os.path.join(test_dataloader.dataset._annopath)
    imagesetfile = os.path.join(test_dataloader.dataset._imgsetpath%test_dataloader.dataset.imgset)
    cachedir = os.path.join(
        Config.testresults_root             # cache /data-nbd/wuxc/111111/voc
    )
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True                    # if year < 2010 else False
    # print("Eval IoU : {:.2f}".format(iou))
    if output_dir is not None and not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(test_dataloader.dataset.CLASSES_NAME):
        if cls == "__background__":
            continue
        filename = get_voc_results_file_template().format(cls)
        rec, prec, ap = voc_eval(
            filename,
            annopath,
            imagesetfile,
            cls,
            cachedir,
            ovthresh=iou,
            use_07_metric=use_07_metric,
        )
        aps += [ap]
        if iou == 0.5:
            print("AP = {:.4f} for {}".format(ap, cls))
        if output_dir is not None:
            with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
    
    if iou == 0.5:
        print("Mean AP = {:.4f}".format(np.mean(aps)))
    return np.mean(aps)

def convert_to_voc_format(conf_pred, class_pred, bbox_pred, img_id):
    """
        
    Args:
        bbox_pred: numpy(nums_pic,4)
    Returns:
        dict
        {
            0 : (bbox, cls, conf)    np.array((n,4),(n, ),(n, ))
            ...
            ...
            4952 : (bbox, cls, conf) np.array((n,4),(n, ),(n, ))
        }
    """
    predictions = {}
    for (conf, cls, bbox, ids) in zip(
        conf_pred, class_pred, bbox_pred, img_id    #for loop n pics
    ):
        if bbox is None:
            predictions[int(ids)] = (None, None, None)
            continue
        predictions[int(ids)] = (bbox, cls, conf)
    return predictions