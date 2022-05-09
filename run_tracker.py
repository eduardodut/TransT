from audioop import reverse
from turtle import forward
from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
from pysot_toolkit.bbox import get_axis_aligned_bbox
import supervisely_lib as sly
import torch
import pandas as pd
from dataset import Tset_Supervisely
import numpy as np
import cv2
import os
import multiprocessing as mp
import json
from glob import glob

def init_tracker(bbox,frame,tracker):
    x1,y1,x2,y2 = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
    cx, cy, w, h = get_axis_aligned_bbox(np.array([x1,y1,x2-x1,y2-y1]))
    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
    init_info = {'init_bbox': gt_bbox_}
    tracker.initialize(frame, init_info)
    
def _track_sequence(sequence,model_pth):
    
    net = NetWithBackbone(net_path=model_pth, use_gpu=True)
    tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)
    output = []
    classe = ''
    for linha in sequence.to_dict(orient="records"):
        img = cv2.cvtColor(cv2.imread(linha['img_path']), cv2.COLOR_RGB2BGR)
        if not np.isnan(linha['confidence']):
            classe = linha['class']
            init_tracker(linha,img,tracker)
            output.append(linha)
        else:            
            outputs = tracker.track(img)
            x1,y1,x2,y2 = outputs['target_bbox']
            x2 = x1+x2
            y2 = y1+y2
            _out = linha.copy()
            _out['confidence'] = np.nan
            _out['class'] = classe
            _out['x_min'], _out['y_min'], _out['x_max'], _out['y_max'] = x1,y1,x2,y2
            output.append(_out)
    return pd.DataFrame(output)   

def add_back_sequence(sequence):
    sequence.sort_values('img_name',inplace=True)
    sequence['forward_sequence'] = sequence['confidence']
    sequence['forward_sequence'].fillna(method='ffill',inplace=True)
    forward_sequences  = sequence.loc[~sequence['forward_sequence'].isna()]
    backward_sequences = sequence.loc[sequence['forward_sequence'].isna()]
    if backward_sequences.shape[0] > 0:
        backward_sequences = pd.concat([backward_sequences,forward_sequences.iloc[[0]]]).iloc[::-1]
        forward_sequences = pd.concat([forward_sequences,backward_sequences])
    
    return forward_sequences.drop("forward_sequence", axis=1)
    
def track_sequence(sequence, sequence_img_root,model_pth,extension='.jpg'):
    torch.cuda.empty_cache()
    sequence.sort_values("img_name",ascending=True,inplace=True)
    img_paths = pd.DataFrame(glob(os.path.join(sequence_img_root,f"*.{extension.replace('.','')}")), columns=['img_path'])
    img_paths['img_name'] = img_paths['img_path'].apply(os.path.basename)
    img_paths['seq_label'] = sequence_img_root.split(os.sep)[-2]
    sequence = img_paths.merge(sequence,on=['seq_label','img_name'],how='left')
    
    sub_sequences = add_back_sequence(sequence)
    out = _track_sequence(sub_sequences,model_pth)
    out.drop('img_path',axis=1,inplace=True)
    out.drop_duplicates(subset=['seq_label', 'img_name'], inplace=True)
    out.sort_values('img_name',inplace=True)
    return out
def main(anchors_obj_path, model_pth,image_root,output_path,min_confidence,extension='.jpg'):
    image_project = pd.read_csv(anchors_obj_path)
    index = image_project.dropna().loc[image_project.dropna().confidence < min_confidence].index
    image_project.drop(index, inplace=True)
    output_name = os.path.basename(anchors_obj_path).replace(".csv","")+"__"+os.path.basename(model_pth).split(".")[0]
    os.makedirs(output_path,exist_ok=True)
    output_path = os.path.join(output_path,output_name+".csv")
    args = []
    out = []
    no_anchors = []
    for seq_label in image_project.seq_label.unique():
        sequence = image_project.loc[image_project.seq_label == seq_label]
        sequence_img_root = os.path.join(image_root,seq_label,'img')
        if sequence.dropna().shape[0] > 0:
            arg = [sequence, sequence_img_root,model_pth,extension]
            args.append(arg)
            track_sequence(*arg)
            print
        else:
            no_anchors.append(sequence)
    pool = mp.Pool(2)
    out = pool.starmap(track_sequence,args)
    pool.close()
    pool.join()
    out = pd.concat(out)
    if len(no_anchors) > 0:
        no_anchors =  pd.concat(no_anchors)
        out = pd.concat([out, no_anchors])
    out.to_csv(output_path,index=False)
    
if __name__ == "__main__":
    # anchors_obj_path = "tset/3_object_detection/ground_truth/lasot_person__0.01__ground_truth.csv"
    anchors_obj_path = "tset/3_object_detection/ground_truth/Tset_images__ground_truth.csv"
    # image_root = "/run/media/eduardo/SSD/Unifall/dataset/LaSOT/lasot_person"
    image_root = "/run/media/eduardo/SSD/Unifall/dataset/T-set/projects/Tset_images"
    
    model_path = "/run/media/eduardo/SSD/Unifall/modelos/TransT/transt.pth"
    output_path = "tset/4_track/output"
    
    extension = 'jpeg'
    min_confidence = 0.98
    main(anchors_obj_path, model_path,image_root,output_path,min_confidence,extension)