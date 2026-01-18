import cv2

from src.face_detection import FaceDetector
from src.landmarks import FaceLandmarkDetector
from src.ear import eye_aspect_ratio


def main():
    # Initialize webcam (macOS-safe backend)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Initialize detectors
    face_detector = FaceDetector()
    landmark_detector = FaceLandmarkDetector()

    # EAR configuration (STEP 2 you asked about)
    EAR_THRESHOLD = 0.25
    DROWSY_FRAMES = 20
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # -------- FACE DETECTION --------
        faces = face_detector.detect(frame)
        for face in faces:
            h, w, _ = frame.shape
            x = int(face.xmin * w)
            y = int(face.ymin * h)
            width = int(face.width * w)
            height = int(face.height * h)

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

        # -------- LANDMARKS + EAR --------
        landmarks = landmark_detector.detect(frame)

        if landmarks:
            h, w, _ = frame.shape

            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Drowsiness logic
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                frame_counter = 0

            if frame_counter >= DROWSY_FRAMES:
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            # Display EAR value
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (30, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        cv2.imshow("Driver Drowsiness Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
