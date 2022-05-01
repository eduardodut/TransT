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
def _track_sequences(sequence,net,ds):
    
    x1,y1,x2,y2 = sequence[2]
    cx, cy, w, h = get_axis_aligned_bbox(np.array([x1,y1,x2-x1,y2-y1]))
    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]

    init_info = {'init_bbox': gt_bbox_}
    tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)
    init_frame = sequence[3]
    final_frame = sequence[4]
    is_reversed = sequence[4] < sequence[3]
    if is_reversed:
        frame_range = list(reversed(range(final_frame,init_frame+1)))
    else:
        frame_range = list(range(init_frame,final_frame+1))
    img_list = [cv2.cvtColor(cv2.imread(frame), cv2.COLOR_RGB2BGR) for frame in ds.iloc[frame_range]['img']]
    tracker.initialize(img_list[0], init_info)
    bbox_list = [sequence[2].tolist()]
    for img in img_list[1:]:
        outputs = tracker.track(img)
        x1,y1,x2,y2 = outputs['target_bbox']
        x2 = x1+x2
        y2 = y1+y2
        bbox_list.append([x1,y1,x2,y2])
    if is_reversed:
        bbox_list = list(reversed(bbox_list))[:-1]
        
        init_frame = final_frame
    # sequence_bbox_list += bbox_list
    return init_frame, bbox_list
    print

def filter_annotation_label(ann_list,annotation_label,min_confidence):
    if len(ann_list) == 0:
        return []
    new_dicts = []
    for k, ann in enumerate(ann_list):
        new_dict = {}
        new_dict['index'] = k
        for tag in ann['tags']:
            new_dict[tag['name']] = tag['value']
           
        new_dicts.append(new_dict)
    confidence = min_confidence
    index = -1
    has_confidence = False
    if len(new_dicts) > 0:
        for new_dict in new_dicts:
            if (index > -1) and (not has_confidence):
                break
            if new_dict['source'] == annotation_label:
                if 'confidence' in new_dict.keys():
                    has_confidence = True
                if has_confidence and (new_dict['confidence'] >= confidence):
                    index = new_dict['index']
                    confidence = new_dict['confidence']
                else:
                    index = new_dict['index']
                   
    if index > -1:
        return np.array(ann_list[index]['points']['exterior']).reshape(-1)
    else:
        return []
    
def track_sequence(model_pth,ds,annotation_label,min_confidence):
        out = {}
        out['init_frame'] = []
        out['bbox_list']  = []
        net = NetWithBackbone(net_path=model_pth, use_gpu=True)
        total_frames = ds.shape[0]
        ds.sort_values('img',inplace=True,ascending=True)
        ds.reset_index(drop=True, inplace=True)
        ds['bbox'] = ds['ann'].apply(lambda x: x['objects'])
        
        ds['bbox'] = ds['bbox'].apply(lambda x: filter_annotation_label(x,annotation_label,min_confidence ))
        ds['init_frame'] = ds.index
        ds = ds[['seq_label','img','bbox','init_frame']]
        
        
        forward_sequences = ds.loc[ds.bbox.apply(lambda x: len(x)>0)]
        forward_sequences['final_frame'] = forward_sequences['init_frame'].shift(-1).fillna(total_frames).astype(int) -1 

        sequences = forward_sequences
        
        backward_sequence = ds.loc[ds.bbox.apply(lambda x: len(x)>0)]
        backward_sequence['final_frame'] = backward_sequence['init_frame'].shift(1).fillna(-1).astype(int) + 1
        if backward_sequence.shape[0]>0:
            backward_sequence = backward_sequence.iloc[[0]]
            b_init_frame = int(backward_sequence.init_frame.values)
            b_final_frame = int(backward_sequence.final_frame.values)
            if b_init_frame != b_final_frame:
                sequences = pd.concat([backward_sequence,sequences],ignore_index=True)
        # backward_init_frame, backward_bbox_list = _track_sequences(backward_sequences,net,ds)
        if sequences.shape[0] > 0:
            _out = sequences.apply(lambda x: _track_sequences(x,net,ds),axis=1)
            # _out = list(_out.apply(pd.DataFrame))
            # forward_init_frame, forward_bbox_list   = _track_sequences(forward_sequences,net,ds)
            # backward_init_frame, backward_bbox_list   = _track_sequences(backward_sequence,net,ds)
            for o in _out:
                out['init_frame'].append(o[0])
                out['bbox_list']+= o[1]
            # out['backward_init_frame'] = backward_init_frame 
            # out['backward_bbox_list'] = backward_bbox_list 
            print
        return out
def track_dataset(ds, dataset,model_pth,output_path,annotation_label,min_confidence):
    torch.cuda.empty_cache()
    out = track_sequence(model_pth,ds,annotation_label,min_confidence)
    if len(out['bbox_list']) > 0:
        ds = ds.iloc[out['init_frame'][0]:]
        out_file = []
        for frame, bbox in zip(ds.itertuples(index=False),out['bbox_list']):
            x1,y1,x2,y2 = bbox
            img_name = os.path.basename(frame[3])
            ann_path = dataset.get_ann_path(img_name)
            out_file.append([ann_path,x1,y1,x2,y2])
            print
        out_file = pd.DataFrame(out_file, columns=['ann_path','x_min','y_min','x_max','y_max'])
        out_file[['x_min','y_min','x_max','y_max']] = out_file[['x_min','y_min','x_max','y_max']].astype(float).round(2)
        out_file.to_csv(output_path,index=False)
        torch.cuda.empty_cache()
def main(dataset_path, model_pth,annotation_label,output_path,min_confidence):
    image_project = Tset_Supervisely(dataset_path)
    image_project.load_anns(12)
    output_name = os.path.basename(dataset_path)+"__"+annotation_label
    output_path = os.path.join(output_path,output_name)
    os.makedirs(output_path,exist_ok=True)
    args = []
    for seq_label in image_project.samples.seq_label.unique():
        dataset = image_project.project.datasets.get(seq_label)
        ds = image_project.filter_by_seq(seq_label).samples
        _output_path = os.path.join(output_path,f"{seq_label}.csv")
        if not os.path.isfile(_output_path):
            args.append([ds, dataset,model_pth,_output_path,annotation_label,min_confidence])
            # track_dataset(ds, dataset,model_pth,_output_path,annotation_label,min_confidence)
    pool = mp.Pool(2)
    pool.starmap(track_dataset,args)
    pool.close()
    pool.join()
    print
       
    



if __name__ == "__main__":
    # image_dataset_path = "/run/media/eduardo/SSD/Unifall/dataset/T-set/projects/Tset_images"
    image_dataset_path = "/run/media/eduardo/SSD/Unifall/dataset/LaSOT/lasot_person"
    model_path = "/run/media/eduardo/SSD/Unifall/modelos/TransT/transt.pth"
    annotation_label = '0.01__ground_truth'
    output_path = "output"
    min_confidence = 0.98
    main(image_dataset_path, model_path,annotation_label,output_path,min_confidence)