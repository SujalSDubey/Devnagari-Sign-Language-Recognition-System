import pickle
import sys
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.stdout.reconfigure(encoding='utf-8')

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False,  # Set to False for continuous tracking
                       max_num_hands=2,  # Detect up to two hands
                       min_detection_confidence=0.3)

# English labels
labels_dict = {
    0: 'Hello',
    1: 'Home',
    2: 'Good',
    3: 'Easy',
    4: 'You',
    5: 'Me',
    6: 'Food',
    7: 'Place',
    8: 'Practice',
    9: 'Sorry',
    10: 'Strong'
}

# Commenting out font-related parts
# from PIL import ImageFont, ImageDraw, Image
# font_path = "D:/@VIT/VIT-2-2/AI/AIPRO/sign-language-detector-python-master/sign-language-detector-python-master/NotoSans-Italic-VariableFont_wdth,wght.ttf"  # Change this to your font file path
# font_size = 36
# font = ImageFont.truetype(font_path, font_size)

while True:
    data_list = []  # List to store data for both hands
    x_combined = []  # Combined x coordinates for both hands
    y_combined = []  # Combined y coordinates for both hands
    predictions = []  # List to store predictions for both hands

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Draw landmarks for each hand
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Get normalized coordinates for each hand separately
            x_list = []
            y_list = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_list.append(x)
                y_list.append(y)

            # Process coordinates and make predictions for each hand
            data_aux = []
            for x_, y_ in zip(x_list, y_list):
                data_aux.append(x_ - min(x_list))
                data_aux.append(y_ - min(y_list))
                x_combined.append(x_)
                y_combined.append(y_)

            data_list.append(data_aux)

        # Pad sequences to a fixed length
        max_sequence_length = 100  # Choose an appropriate maximum length
        padded_data = pad_sequences(data_list, maxlen=max_sequence_length, padding='post', truncating='post', dtype='float32')

        # Make predictions for padded data
        if len(padded_data) > 0:
            predictions = model.predict(padded_data)

        # Display predictions
        for i, prediction in enumerate(predictions):
            predicted_label = labels_dict[int(prediction)]
            x1 = int(min(x_combined) * W) - 10
            y1 = int(min(y_combined) * H) - 10
            x2 = int(max(x_combined) * W) - 10
            y2 = int(max(y_combined) * H) - 10

            # Commenting out PIL drawing parts
            # pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # draw = ImageDraw.Draw(pil_img)
            # draw.text((x1, y1 - font_size), predicted_label, font=font, fill=(0, 0, 0))
            # frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
