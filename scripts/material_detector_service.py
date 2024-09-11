#!/home/user/pel_ws/pel_venv/bin/python
import sys
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from detection_msgs.msg import BoundingBoxes, BoundingBox  # Import BoundingBoxes and BoundingBox message
from pel_ros.srv import ImageDetection, ImageDetectionResponse  # Import the service definition
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Loading vision transformer model fine-tuned on the MINC2500 subset of the MINC dataset
processor = AutoImageProcessor.from_pretrained("ioanasong/vit-MINC-2500")
model = AutoModelForImageClassification.from_pretrained("ioanasong/vit-MINC-2500") 

class YoloVitMaterialDetectorService:
    def __init__(self):
        self.bridge = CvBridge()
        self.labels_materials = ['brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair',
                                 'leather', 'metal', 'mirror', 'other', 'painted', 'paper', 'plastic', 'polishedstone',
                                 'skin', 'sky', 'stone', 'tile', 'wallpaper', 'water', 'wood']
        self.yolo_model = YOLO('yolov10s.pt')
        self.vit_model = model
        self.vit_processor = processor
        self.service = rospy.Service('material_detection_service', ImageDetection, self.handle_material_detection)  # ROS service
        rospy.loginfo("material_detection_service is ready.")
        
    def handle_material_detection(self, req):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return ImageDetectionResponse()

        # YOLO detection
        results = self.yolo_model(cv_image)
        
        # Create BoundingBoxes message
        bounding_boxes_msg = BoundingBoxes()
        bounding_boxes_msg.header.stamp = rospy.Time.now()
        bounding_boxes_msg.header.frame_id = req.image.header.frame_id
        bounding_boxes_msg.image_header = req.image.header

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                crop = cv_image[y1:y2, x1:x2]
                
                # Vision Transformer inference
                inputs = self.vit_processor(images=crop, return_tensors="pt")
                outputs = self.vit_model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
                material = self.labels_materials[predicted_class]
                
                # Create BoundingBox message for each detected object
                bbox_msg = BoundingBox()
                bbox_msg.Class = result.names[int(classes[i])]
                bbox_msg.additional_label = material
                bbox_msg.probability = float(confidences[i])
                bbox_msg.xmin = x1
                bbox_msg.ymin = y1
                bbox_msg.xmax = x2
                bbox_msg.ymax = y2

                # Append the bounding box to bounding_boxes_msg
                bounding_boxes_msg.bounding_boxes.append(bbox_msg)
        
        # Return the ImageDetectionResponse populated with the BoundingBoxes message
        return ImageDetectionResponse(objects=bounding_boxes_msg)

if __name__ == '__main__':
    rospy.init_node('yolo_vit_mat_detector_service')
    detector = YoloVitMaterialDetectorService()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv2.destroyAllWindows()
