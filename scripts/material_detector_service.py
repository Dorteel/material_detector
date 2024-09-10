#!/home/user/pel_ws/pel_venv/bin/python
import sys
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from vit_inference.srv import MaterialDetection, MaterialDetectionResponse  # Import the service definition
import numpy as np

# Loading vision transformer model fine-tuned on the MINC2500 subset of the MINC dataset
processor = AutoImageProcessor.from_pretrained("ioanasong/vit-MINC-2500")
model = AutoModelForImageClassification.from_pretrained("ioanasong/vit-MINC-2500") 

class YoloVitMaterialDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.labels_materials = ['brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair',
                             'leather', 'metal', 'mirror', 'other', 'painted', 'paper', 'plastic', 'polishedstone',
                             'skin', 'sky', 'stone', 'tile', 'wallpaper', 'water', 'wood']
        self.yolo_model = YOLO('yolov10s.pt')
        self.vit_model = model
        self.vit_processor = processor
        self.service = rospy.Service('material_detection_service', MaterialDetection, self.handle_material_detection)  # ROS service
        print("Service initialized")

    def handle_material_detection(self, req):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "passthrough")
            cv_image = np.array(cv_image, copy=True)
        except CvBridgeError as e:
            rospy.logerr(e)
            return MaterialDetectionResponse("", "", 0.0, 0, 0, 0, 0)

        # YOLO detection
        results = self.yolo_model(cv_image)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                crop = cv_image[y1:y2, x1:x2]
                
                # Vision Transformer inference
                inputs = self.vit_processor(images=crop, return_tensors="pt")
                outputs = self.vit_model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
                confidence = outputs.logits.softmax(-1).max().item()
                
                # Returning the first detected object for simplicity
                det_msg = MaterialDetectionResponse()
                det_msg.object_class = result.names[int(result.boxes.cls[0])]
                det_msg.confidence = confidence
                det_msg.x = x1
                det_msg.y = y1
                det_msg.width = x2 - x1
                det_msg.height = y2 - y1
                det_msg.material = self.labels_materials[predicted_class]

                return det_msg
        
        # If no objects are detected, return an empty response
        return MaterialDetectionResponse("", "", 0.0, 0, 0, 0, 0)

if __name__ == '__main__':
    rospy.init_node('yolo_vit_mat_detector_service')
    detector = YoloVitMaterialDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
