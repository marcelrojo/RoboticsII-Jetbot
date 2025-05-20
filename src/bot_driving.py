import cv2
import onnxruntime as rt

import os
import time

from pathlib import Path
import yaml
import numpy as np

from PUTDriver import PUTDriver, gstreamer_pipeline


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img_size = 64
        
        img = cv2.resize(img, (img_size, img_size))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        img_blurred = cv2.GaussianBlur(img_clahe, (3, 3), 0)
        
        img_crop = img_blurred[img_size//2:img_size, 0:img_size]
        img_crop = cv2.resize(img_crop, (img_size, img_size))
        
        img_normalized = img_crop.astype(np.float32) / 255.0
        img_normalized = np.expand_dims(img_normalized, axis=-1)

        return np.array([img_normalized])

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        ##TODO: prepare your outputs

        return detections

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        #assert inputs.dtype == np.float32
        #assert inputs.shape == (1, 1, 64, 64)
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        #assert outputs.dtype == np.float32
        #assert outputs.shape == (2,)
        #assert outputs.max() < 1.0
        #assert outputs.min() > -1.0

        return outputs


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print(f'No camera')
        return
    
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)
    frame_count = 0
    
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break
        if frame_count % 60 == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"frame_{frame_count}_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, image)

        forward, left = ai.predict(image)
        frame_count += 1

if __name__ == '__main__':
        main()
