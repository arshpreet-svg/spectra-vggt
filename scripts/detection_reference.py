"""Independent YOLOv8 ONNX video reference for PipeGen to inspect.

Requires a real standard YOLOv8 detection ONNX export (no end-to-end NMS).
This runs OpenCV DNN, NOT PipeGen/TensorRT. Never presented as PipeGen evidence.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np


def detect(net, frame, labels, threshold):
    height,width=frame.shape[:2]
    scale=min(640/width,640/height)
    resized=cv2.resize(frame,(round(width*scale),round(height*scale)))
    left=(640-resized.shape[1])//2;top=(640-resized.shape[0])//2
    canvas=np.full((640,640,3),114,dtype=np.uint8)
    canvas[top:top+resized.shape[0],left:left+resized.shape[1]]=resized
    net.setInput(cv2.dnn.blobFromImage(canvas,1/255,(640,640),swapRB=True,crop=False))
    output=net.forward()
    if output.ndim!=3 or output.shape[0]!=1 or output.shape[1]!=4+len(labels):
        raise ValueError(f'Expected YOLOv8 [1,{4+len(labels)},N], got {output.shape}; verify model/parser compatibility')
    boxes,scores,classes=[],[],[]
    for row in output[0].T:
        if not np.isfinite(row).all():continue
        category=int(np.argmax(row[4:]));score=float(row[4+category])
        if score<threshold:continue
        cx,cy,bw,bh=map(float,row[:4])
        x1=max(0.,min(width,(cx-bw/2-left)/scale));y1=max(0.,min(height,(cy-bh/2-top)/scale))
        x2=max(0.,min(width,(cx+bw/2-left)/scale));y2=max(0.,min(height,(cy+bh/2-top)/scale))
        if x2<=x1 or y2<=y1:continue
        boxes.append([x1,y1,x2-x1,y2-y1]);scores.append(score);classes.append(category)
    detections=[]
    # Class-specific NMS so overlapping objects of different classes are retained.
    for category in sorted(set(classes)):
        indices=[i for i,c in enumerate(classes) if c==category]
        selected=cv2.dnn.NMSBoxes([boxes[i] for i in indices],[scores[i] for i in indices],threshold,.45)
        for local in np.asarray(selected).reshape(-1):
            index=indices[int(local)];x,y,w,h=boxes[index]
            detections.append({'label':labels[category],'confidence':scores[index],'bbox_xyxy':[x,y,x+w,y+h]})
    return detections


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True,help='JSON array of class labels in model order')
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-frames',type=int,default=100)
    parser.add_argument('--confidence',type=float,default=.35)
    args=parser.parse_args()
    if args.max_frames<1 or not 0<=args.confidence<=1:parser.error('Invalid frame limit or confidence')
    labels=json.loads(args.labels.read_text())
    if not isinstance(labels,list) or not labels or not all(isinstance(x,str) and x for x in labels):
        parser.error('labels must be a nonempty JSON array of strings')
    net=cv2.dnn.readNetFromONNX(str(args.model))
    capture=cv2.VideoCapture(str(args.video))
    if not capture.isOpened():raise ValueError('Cannot open video')
    fps=capture.get(cv2.CAP_PROP_FPS)
    if fps<=0:raise ValueError('Video FPS unavailable')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with args.output.open('w') as stream:
            for index in range(args.max_frames):
                ok,frame=capture.read()
                if not ok:break
                start=time.perf_counter();detections=detect(net,frame,labels,args.confidence)
                event={'schema_version':1,'producer':'opencv-reference','frame_index':index,
                       'timestamp_seconds':index/fps,'image_size':[frame.shape[1],frame.shape[0]],
                       'detections':detections,'inference_seconds':time.perf_counter()-start}
                stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush()
    finally:capture.release()

if __name__=='__main__':main()
