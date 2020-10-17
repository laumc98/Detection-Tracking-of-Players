#!/usr/bin/python
# -*- coding: utf-8 -*-

# import the necessary packages
from video import FileVideoStream
from imutils.video import FPS
from centroidtracker import CentroidTracker
import numpy as np
import argparse
import imutils
import cv2
import detectron2
import sys
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou


area0 = []; area1 = []; area2 = []; area3=[]; area4=[]
def area(coor0,coor1,coor2,coor3,coor4):
    M0 = cv2.moments(np.float32(coor0))
    a0 = int(M0['m00'])
    area0.append(a0)
    #print(type(area0))
    #print("Area ID0 = ",area0)
    M1 = cv2.moments(np.float32(coor1))
    a1 = int(M1['m00'])
    area1.append(a1)
    #print("Area ID1 = ",area1)
    M2 = cv2.moments(np.float32(coor2))
    a2 = int(M2['m00'])
    area2.append(a2)
    #print("Area ID2 = ",area2)
    M3 = cv2.moments(np.float32(coor3))
    a3 = int(M3['m00'])
    area3.append(a3)
    #print("Area ID3 = ",area3)
    M4 = cv2.moments(np.float32(coor4))
    a4 = int(M4['m00'])
    area4.append(a4)
    #print("Area ID4 = ",area4)

    return area0,area1,area2,area3,area4
        
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True,
                help='path to input video file')
ap.add_argument('-o', '--output', type=str,
                help='path to optional output video file')
ap.add_argument("-c", "--confidence", type=float, default=0.90,
        help="minimum probability to filter weak detections")
ap.add_argument("-t", "--starttime", type=float, default=0,
        help="Start time for the test")
args = vars(ap.parse_args())

# start the file video stream thread and allow the buffer to start to fill
print ('[INFO] starting video file thread...')
fvs = FileVideoStream(args['video']).start()

writer = None
time.sleep(1.0)

#Define the predictor and the centroidtracker
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
                    ))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = \
    model_zoo.get_checkpoint_url('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
                                 )
predictor = DefaultPredictor(cfg)
ct = CentroidTracker()

# Define the frame per second from the video file
framepersec = fvs.fps()

minu=0
con=0
cont=0
tiempo=[]

coor0= [];coor1= [];coor2= [];coor3= [];coor4= []
Xid0= [];Yid0= [];Xid1= [];Yid1= [];Xid2= [];Yid2= [];Xid3= [];Yid3= [];Xid4= [];Yid4= []
plt.rcParams.update({'figure.max_open_warning': 0})

while fvs.more():

    frame = fvs.read()
    if frame is None:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if args['output'] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args['output'], fourcc, framepersec,
                                 (frame.shape[1], frame.shape[0]), True)
    frames=round(framepersec,3)
    outputs = predictor(frame)

    instances = outputs['instances']
    scores = instances.scores.cpu().numpy()
    cuadra = []
    rects = []
    #print ('Numero de Identificados: ', len(instances))

    i = 0
    con +=1
    cont +=1
    # loop over the detections
    for i in range(0, len(instances)):
        confidence = scores[i]
        if confidence > args["confidence"]:

            # Boxes
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS,
                                    BoxMode.XYWH_ABS)
            boxes = boxes.tolist()
            # print('Coordenadas Recuadros')
            # print(str(i)+'-',boxes[i])

            # Keypoints
            has_keypoints = instances.has('pred_keypoints')
            if has_keypoints:
                keypoints = instances.pred_keypoints.cpu().numpy()
                keypoints = keypoints[:, :, :2]
                keypoints = keypoints.tolist()
                # print('Keypoints identificados')
                # print(str(i)+'-',keypoints[i])

            # Tracking
            (x, y, w, h) = boxes[i]
            cuadra.append([x, y, x + w, y + h])
            box = cuadra[i] * np.array([1, 1, 1, 1])
            rects.append(box.astype('int'))
            (startX, startY, endX, endY) = box.astype('int')
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,
                          255, 0), 2)
    
    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)
    Idcen, Idrec = objects
    #print("lista completa",objects)

    #Bounding box coordinates for each ID
    for ju in range(1):
        try:
            rec0 = Idrec[0]
            coor0.append(rec0)
            X0=rec0[2]
            Xid0.append(X0)
            Y0=rec0[3]
            Yid0.append(Y0)
            #print("Coordenadas ID0: ",coor0)
            #print("Cordenada X0: ",Xid0)
            #print("Cordenada Y0: ",Yid0)
            rec1 = Idrec[1]
            coor1.append(rec1)
            #print(type(coor1))
            X1= rec1[2]
            Xid1.append(X1)
            Y1= rec1[3]
            Yid1.append(Y1)
            #print("Coordenadas ID1: ",coor1)
            #print("Cordenada X1: ",Xid1)
            #print("Cordenada Y1: ",Yid1)
            rec2 = Idrec[2]
            coor2.append(rec2)
            X2=rec2[2]
            Xid2.append(X2)
            Y2=rec2[3]
            Yid2.append(Y2)
            #print("Coordenadas ID2: ",coor2)
            #print("Cordenada X2: ",Xid2)
            #print("Cordenada Y2: ",Yid2)
            rec3 = Idrec[3]
            coor3.append(rec3)
            X3=rec3[2]
            Xid3.append(X3)
            Y3=rec3[3]
            Yid3.append(Y3)
            #print("Coordenadas ID3: ",coor3)
            print("Cordenada X3: ",Xid3)
            #print("Cordenada Y3: ",Yid3)
            rec4 = Idrec[4]
            coor4.append(rec4)
            X4=rec4[2]
            Y4=rec4[3]
            Xid4.append(X4)
            Yid4.append(Y4)
            #print("Coordenadas ID4: ",coor4)
            print("Cordenada X4: ",Xid4)
            #print("Cordenada Y4: ",Yid4)
        except KeyError:
            continue
                    
    for (objectID, centroid) in Idcen.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        coordenada = centroid
        text = 'Id{}'.format(objectID)
        cv2.putText(
            frame,
            text,
            (centroid[0] - 20, centroid[1] - 20),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
            )
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0),
                   -1)
        
    #We define the time for the video
    seg=con/frames
    tiempo_ = cont
    tiempo.append(tiempo_)
    #print("contador",cont)
    #print("tiempo",tiempo)
    #t=np.array(tiempo)
    
    if seg >= 60:
        con=0
        minu +=1

    time = f"{minu:02d}:{int(seg):02d}"
    print(time)

    text2 = 'Tiempo: '+ time

    cv2.putText(
        frame,
        text2,
        (20,30),
        cv2.FONT_HERSHEY_COMPLEX,
        1.2,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
        )

    area0,area1,area2,area3,area4=area(coor0,coor1,coor2,coor3,coor4)

    #Plot Areas Vs Frames
    fig1=plt.figure()
    ax = plt.subplot(111)
    ax.plot(tiempo, area0, color='b', label='Area ID0')
    ax.plot(tiempo, area1, color='g', label='Area ID1')
    ax.plot(tiempo, area2, color='r', label='Area ID2')
    ax.plot(tiempo, area3, color='m', label='Area ID3')
    ax.plot(tiempo, area4, color='y', label='Area ID4')
    plt.title('Areas vs Frames')
    plt.ylabel("Areas")
    plt.xlabel("Numero Frames")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=4)
    fig1.savefig('Areas.png')
    plt.close(fig1)

    #Plot (x,y) players foot Vs Frames
    #fig2=plt.figure()
    #ax0 = fig2.add_subplot(111,projection='3d')
    #ax0.plot_wireframe(np.array([Xid0]),np.array([Yid0]), np.array([tiempo]))
    #ax0.set_title('Desplazamiento en X, Y, Z ID0', color='b')
    #ax0.set_zlabel('Numero de Frames')
    #ax0.set_xlabel('Posicion en x')
    #ax0.set_ylabel('Posicion en y')
    #plt.tight_layout()
    #fig2.savefig('X-Y ID0.png')
    #plt.close(fig2)
    
    #fig3=plt.figure()
    #ax1 = fig3.add_subplot(111,projection='3d')
    #ax1.plot_wireframe(np.array([Xid1]),np.array([Yid1]), np.array([tiempo]))
    #ax1.set_title('Desplazamiento en X, Y, Z ID1', color='g')
    #ax1.set_zlabel('Numero de Frames')
    #ax1.set_xlabel('Posicion en x')
    #ax1.set_ylabel('Posicion en y')
    #plt.tight_layout()
    #fig3.savefig('X-Y ID1.png')
    #plt.close(fig3)
    
    #fig4=plt.figure()
    #ax2 = fig4.add_subplot(111,projection='3d')
    #ax2.plot_wireframe(np.array([Xid2]),np.array([Yid2]), np.array([tiempo]))
    #ax2.set_title('Desplazamiento en X, Y, Z ID2', color='r')
    #ax2.set_zlabel('Numero de Frames')
    #ax2.set_xlabel('Posicion en x')
    #ax2.set_ylabel('Posicion en y')
    #plt.tight_layout()
    #fig4.savefig('X-Y ID2.png')
    #plt.close(fig4)
    
    #fig5=plt.figure()
    #ax3 = fig5.add_subplot(111,projection='3d')
    #ax3.plot_wireframe(np.array([Xid3]),np.array([Yid3]), np.array([tiempo]))
    #ax3.set_title('Desplazamiento en X, Y, Z ID3', color='m')
    #ax3.set_zlabel('Numero de Frames')
    #ax3.set_xlabel('Posicion en x')
    #ax3.set_ylabel('Posicion en y')
    #plt.tight_layout()
    #fig5.savefig('X-Y ID3.png')
    #plt.close(fig5)
    
    #fig6=plt.figure()
    #ax4 = fig6.add_subplot(111,projection='3d')
    #ax4.plot_wireframe(np.array([Xid4]),np.array([Yid4]), np.array([tiempo]))
    #ax4.set_title('Desplazamiento en X, Y, Z ID4', color='y')
    #ax4.set_zlabel('Numero de Frames')
    #ax4.set_xlabel('Posicion en x')
    #ax4.set_ylabel('Posicion en y')
    #plt.tight_layout()
    #fig6.savefig('X-Y ID4.png')
    #plt.close(fig6)
        
    # show the output frame
    cv2.imwrite('Frame.png', frame)

    if writer is not None:
        writer.write(frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
