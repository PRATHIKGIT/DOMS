
import cv2
import numpy as np
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\DIVYANSH\cv\shape_predictor_68_face_landmarks.dat")

while True:
    ret, img = cap.read()
    if not ret:
        break
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray is None:
        break

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        nose = landmarks[27]
        mouth = landmarks[48:68]

        #  head rotation angles using eye and nose landmarks
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)
        nose_center = nose.astype(int)

      
        dX = right_eye_center[0] - left_eye_center[0]
        dY = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dY, dX))

      
        cv2.putText(img, f"Head Rotation: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
        for (x, y) in np.concatenate((left_eye, right_eye, [nose_center], mouth)):
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Frame", img)

cap.release()
cv2.destroyAllWindows()
