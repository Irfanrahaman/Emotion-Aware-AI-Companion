import cv2
from .face_emotion import predict_emotion

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoCamera(object):
    def __init__(self, app_state):
        self.video = cv2.VideoCapture(0)
        self.app_state = app_state

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_emotion = "Neutral"
        for (x, y, w, h) in faces:
            face_roi_color = frame[y:y+h, x:x+w]
            emotion = predict_emotion(face_roi_color)
            detected_emotion = emotion

            cv2.rectangle(frame, (x, y), (x+w, y+h), (71, 194, 13), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (71, 194, 13), 2)
        
        self.app_state['face_emotion'] = detected_emotion
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()