import numpy as np
import cv2

def plot_bbox(img, facepoints, name, prob=''):
    bbox = facepoints['box']
    x, y, w, h = bbox
    x1, y1 = abs(x), abs(y)
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
    cv2.putText(img, name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    if prob != '':
        cv2.putText(img, str(prob), (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)    
    return img

def plot_faces(img, facepoints, names, probs=[]):
    probs = np.round(probs, 2)
    for idx, face in enumerate(facepoints):
        plot_bbox(img, facepoints[idx], names[idx], probs[idx])
    return img