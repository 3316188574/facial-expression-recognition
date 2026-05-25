import cv2
import torch
import numpy as np
from torchvision import transforms

from models.cnn_model import EnhancedCNN


class RealtimeFER:
    def __init__(self, model_path, device='cuda'):
        self.device = device

        # 加载模型
        self.model = EnhancedCNN(num_classes=7).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 从checkpoint中获取类别名称（如果没有则使用默认）
        self.emotion_labels = checkpoint.get('class_names',
                                             ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])

        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 颜色映射
        self.colors = {
            'Angry': (0, 0, 255),
            'Disgust': (0, 255, 255),
            'Fear': (255, 0, 255),
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Surprise': (255, 255, 0),
            'Neutral': (255, 255, 255)
        }

        # 预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict_emotion(self, face_roi):
        """预测表情"""
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return self.emotion_labels[predicted.item()], confidence.item()

    def run(self):
        """运行实时检测"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("实时表情识别已启动... 按 'q' 退出")
        print(f"识别类别: {self.emotion_labels}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                emotion, confidence = self.predict_emotion(face_roi)

                color = self.colors.get(emotion, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('FER System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recognizer = RealtimeFER(args.model_path, device)
    recognizer.run()


if __name__ == '__main__':
    main()