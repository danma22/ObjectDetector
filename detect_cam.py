from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
	parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
	parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
	parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
	parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
	parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
	parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
	parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
	parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
	parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
	opt = parser.parse_args()
	print(opt)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	os.makedirs("output", exist_ok=True)

	# Set up model
	model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

	if opt.weights_path.endswith(".weights"):
		# Load darknet weights
		model.load_darknet_weights(opt.weights_path)
	else:
		# Load checkpoint weights
		model.load_state_dict(torch.load(opt.weights_path))

	model.eval()  # Set in evaluation mode

	classes = load_classes(opt.class_path)  # Extracts class labels from file

	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	imgs = []  # Stores image paths
	img_detections = []  # Stores detections for each image index
	
	camera = cv2.VideoCapture(0)
	
	colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
	
	while camera:
		ret, frame = camera.read()

		frame = cv2.resize(frame, (1000, 800), interpolation=cv2.INTER_CUBIC)
		#LA imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
		
		imgTensor = transforms.ToTensor()(frame)
		imgTensor, _ = pad_to_square(imgTensor, 0)
		imgTensor = resize(imgTensor, 416)
		imgTensor = imgTensor.unsqueeze(0)
		frameTensor = Variable(imgTensor.type(Tensor))
		
		
		with torch.no_grad():
			detections = model(frameTensor)
			detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

	
		for detection in detections:
			# Draw bounding boxes and labels of detections
			if detection is not None:
				# Rescale boxes to original image
				detection = rescale_boxes(detection, opt.img_size, frame.shape[:2])
				unique_labels = detection[:, -1].cpu().unique()
				n_cls_preds = len(unique_labels)
				for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:

					box_w = x2 - x1
					box_h = y2 - y1

					color = [int(c) for c in colors[int(cls_pred)]]
					# Create a Rectangle patch
					frame_ = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
					cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
					cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de p

		cv2.imshow('frame', frame)
	
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	camera.release()
	cv2.destroyAllWindows()
