import cv2
import sys
import torch
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
from TRTWrapper import create_model_wrapper
from utils.general import non_max_suppression
import numpy as np
import yaml
import copy

class Colors():
    """Color class."""
    def __init__(self):
        hex = ('B55151', 'FF3636', 'FF36A2', 'CB72A2', 'EC3AFF', '3B1CFF', '7261E1', '6991BF', '00B1BD', '00BD8B',
               '00DA33', 'BEEF4D', '8B8B8B', 'FFB300', '7F5903', '411C06', '795454', '495783', '624F70', '7A7D62')
        self.palette = [self.hextorgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, m, bgr=False):
        c = self.palette[int(m) % self.n]
        return (c[2], c[1], c[0])

    @staticmethod
    def hextorgb(hex):
        return tuple(int(hex[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def preprocess_image(model, raw_bgr_image):
    input_size = model.input_spec()[0][-2:] # h,w = 640,640
    original_image = raw_bgr_image
    origin_h, origin_w, origin_c = original_image.shape
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # Calculate width and height and paddings
    r_w = input_size[1] / origin_w
    r_h = input_size[0] / origin_h
    if r_h > r_w:
        tw = input_size[1]
        th = int(r_w *  origin_h)
        tx1 = tx2 = 0
        ty1 = int((input_size[0] - th) / 2)
        ty2 = input_size[0] - th - ty1
    else:
        tw = int(r_h * origin_w)
        th = input_size[0]
        tx1 = int((input_size[1] - tw) / 2)
        tx2 = input_size[1] - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    preprocessed_image = np.ascontiguousarray(image)
    return preprocessed_image

def print_result(result_image, result_label, classes):
    detections = non_max_suppression(torch.Tensor(result_label), 0.25, 0.45, max_det=1000)
    detections = detections[0]
    num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = detections.shape[0], detections[:, 0:4], detections[:, 4:5], detections[:, 5:]
    
    colors = Colors()
    result_image = np.squeeze(result_image)
    result_image = result_image.astype(np.uint8)
    print("--------------------------------------------------------------")
    for i in range(int(num_detections)):
        detected = str(classes[int(nmsed_classes[i][0])]).replace('‘', '').replace('’', '')
        confidence_str = str(nmsed_scores[i][0])
        # unnormalize depending on the visualizing image size
        x1 = int(nmsed_boxes[i][0])
        y1 = int(nmsed_boxes[i][1])
        x2 = int(nmsed_boxes[i][2])
        y2 = int(nmsed_boxes[i][3])
        color = colors(int(nmsed_classes[i][0]), True)
        result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        text_size, _ = cv2.getTextSize(str(detected), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        result_image = cv2.rectangle(result_image, (x1, y1-5-text_h), (x1+text_w, y1), color, -1)
        result_image = cv2.putText(result_image, str(detected), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print("Detect " + str(i+1) + "(" + str(detected) + ")")
        print("Coordinates : [{:d}, {:d}, {:d}, {:d}]".format(x1, y1, x2, y2))
        print("Confidence : {:.7f}".format(nmsed_scores[i][0]))
        print("")
    print("--------------------------------------------------------------\n\n")
    return result_image

if __name__ == "__main__":
    print("model Loading...")
    wrapper = create_model_wrapper("./yolov5m.engine", 1, "GPU")
    wrapper.load_model()
    print("model loading complete")
    
    cap = cv2.VideoCapture(0)
    print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
    
    with open('../data/coco.yaml', 'r') as f:
        classes = yaml.safe_load(f)
        classes = classes['names']
    
    while True:
        ret, frame = cap.read()
        if ret:
            img = preprocess_image(wrapper, copy.deepcopy(frame))
            inf_res = wrapper.inference(img)
            res = print_result(frame, inf_res, classes)
            cv2.imshow('frame', res)
            cv2.waitKey(33)
        else :
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    