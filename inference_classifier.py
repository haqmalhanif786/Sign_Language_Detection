# Import necessary libraries
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open the camera
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Hello!', 1: 'I Love You', 2: 'Mother', 3: 'Father', 4:'Friend', 5: 'Yes', 6: 'NO',7: 'Please', 8: 'Thank You!!!'}


while True:
    ret, frame = cap.read()  # Read the frame
    if not ret:
        print("Error: Could not read frame.")
        break  # Exit the loop if the frame was not captured successfully

    # Ensure frame is not None
    if frame is None:
        print("Error: Frame is None.")
        break

    H, W, _ = frame.shape  # Access the shape of the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Make sure data_aux has the right number of features
        if len(data_aux) < 84:
            data_aux += [0] * (84 - len(data_aux))  # Pad if needed
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]  # Trim if too long

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3, cv2.LINE_AA)
        
        # Create a PIL image from the frame
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Define text position
        text_position = (x1, y1 - 50)  # Adjust position as needed

        # Convert back to OpenCV format
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
