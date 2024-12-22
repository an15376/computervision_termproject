import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from camera_objectdetection_ui import Ui_MainWindow
from torchvision import models
import torch.nn as nn

class ObjectDetectionApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ObjectDetectionApp, self).__init__()
        self.setupUi(self)

        # 버튼 연결
        self.DetectButton.clicked.connect(self.toggle_detection)

        # 카메라 초기화
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_feed)
        self.detection_enabled = False

        # YOLO 모델 로드
        self.yolo_model = YOLO('models/strawberry_detection_model_test2.pt')  # YOLOv8 모델 경로

        # CNN 분류 모델 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_model = self.load_cnn_model('models/strawberry_classifier.pth')
        self.cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.class_names = ['ripe strawberry', 'damaged strawberry', 'rotten strawberry', 'green strawberry']

    def load_cnn_model(self, model_path):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 4)  # 4개의 클래스
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def update_camera_feed(self):
        ret, frame = self.camera.read()
        if ret:
            # BGR -> RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.detection_enabled:
                # YOLO로 객체 탐지
                results = self.yolo_model.predict(source=frame)
                detections = results[0]

                # 높은 정확도만 필터링 (예: 0.5 이상의 confidence만 표시)
                filtered_detections = [(box, confidence, cls) for box, confidence, cls in zip(
                    detections.boxes.xyxy.cpu().numpy(),
                    detections.boxes.conf.cpu().numpy(),
                    detections.boxes.cls.cpu().numpy()) if confidence > 0.5]

                # 각 바운딩 박스를 CNN으로 분류
                for box, confidence, cls in filtered_detections:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_img = rgb_frame[y1:y2, x1:x2]

                    # CNN 모델로 클래스 분류
                    classification_result = self.classify_object(cropped_img)
                    label = f"{self.class_names[classification_result]} ({confidence:.2f})"

                    # 바운딩 박스 및 레이블 추가
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # QImage 변환
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # QLabel 업데이트
            self.CameraFeed.setPixmap(QPixmap.fromImage(qt_image).scaled(800, 600, Qt.KeepAspectRatio))

    def classify_object(self, cropped_img):
        # PIL 이미지로 변환 후 CNN 모델에 입력
        pil_image = Image.fromarray(cropped_img)
        input_tensor = self.cnn_transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.cnn_model(input_tensor)
        return torch.argmax(output, dim=1).item()

    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        self.DetectButton.setText("Stop" if self.detection_enabled else "Start Object Detection")

    def closeEvent(self, event):
        self.camera.release()
        self.timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    window.timer.start(30)
    sys.exit(app.exec_())
