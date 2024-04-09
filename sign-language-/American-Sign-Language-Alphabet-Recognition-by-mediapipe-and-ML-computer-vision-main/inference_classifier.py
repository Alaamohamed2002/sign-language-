import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0:'ThumbsDown',1:'One',2:'Three'
               }

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []  # Initialize data_aux for the hand
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Calculate the range of x and y coordinates for normalization
            x_min = min(x_)
            x_max = max(x_)
            y_min = min(y_)
            y_max = max(y_)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                # Normalize x and y coordinates
                data_aux.append((x - x_min) / (x_max - x_min))
                data_aux.append((y - y_min) / (y_max - y_min))

            # Ensure that data_aux has exactly 42 features
            assert len(data_aux) == 42

            # Calculate bounding box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = int(model.predict([np.asarray(data_aux)])[0])  # Convert prediction to integer
            if 0 <= prediction <= 49:
                predicted_character = labels_dict[prediction]
            else:
                predicted_character = "Unknown Gesture"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()