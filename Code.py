import cv2
import mediapipe as mp

# Initialize MediaPipe Hands: Configuration de MediaPipe pour la détection des mains.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Dictionnaire des symboles et leurs signification
finger_symbols = {
    '00000': 'Force',
    '00001': 'Promesse',
    '00100': 'HopHopHop',
    '00111': 'OK',
    '01001': 'Demon',
    '01100': 'Peace',
    '10001': 'Shaka',
    '11001': 'RockNRoll'
}

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

def finger_states_to_string(finger_states):
    # Convertit la liste des états des doigts en chaîne de caractères sans les crochets et les espaces
    return ''.join(str(state) for state in finger_states)

def calculate_bounding_box(landmarks, width, height, padding=2):
    # Calcule un rectangle englobant autour de la main détectée.
    x_min = width
    y_min = height
    x_max = y_max = 0

    for landmark in landmarks:
        x, y = int(landmark.x * width), int(landmark.y * height)
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

    # Adjusting the bounding box with padding
    return max(0, x_min - padding), max(0, y_min - padding), min(width, x_max + padding), min(height, y_max + padding)

def get_finger_states(hand_landmarks, handedness):
    # Détermine si chaque doigt est levé ou baissé.
    thumb_tip_id = mp_hands.HandLandmark.THUMB_TIP.value
    thumb_ip_id = mp_hands.HandLandmark.THUMB_IP.value
    finger_tip_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP.value, mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value,
                      mp_hands.HandLandmark.RING_FINGER_TIP.value, mp_hands.HandLandmark.PINKY_TIP.value]
    finger_states = []

    # Thumb
    if handedness == 'Right':
        thumb_is_open = hand_landmarks.landmark[thumb_tip_id].x > hand_landmarks.landmark[thumb_ip_id].x
    else:  # Left hand
        thumb_is_open = hand_landmarks.landmark[thumb_tip_id].x < hand_landmarks.landmark[thumb_ip_id].x
    finger_states.append(1 if thumb_is_open else 0)

    # Fingers
    for tip_id in finger_tip_ids:
        finger_is_open = hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y
        finger_states.append(1 if finger_is_open else 0)

    return finger_states

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, flipCode = 1) # Retourne l'image horizontalement pour un effet miroir
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame = cv2.resize(frame, (640, 480)) # Redimensionne l'image.
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # Traite l'image avec MediaPipe

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Dessine les landmarks et les connexions de la main.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
             
            # Calculate and draw bounding box with padding
            x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks.landmark, 640, 480, padding=15)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)

            # Compte les doigts levés et obtient l'état des doigts sous forme de chaîne.
            fingers_up = 0
            for id, finger in enumerate(mp_hands.HandLandmark):
                if id == mp_hands.HandLandmark.THUMB_TIP:
                    # Special case for the thumb
                    if hand_landmarks.landmark[id].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
                        fingers_up += 1
                elif id in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]:
                    if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
                        fingers_up += 1
            # Get finger states as a string
            finger_states = get_finger_states(hand_landmarks, handedness_info.classification[0].label)
            finger_states_str = finger_states_to_string(finger_states)
            symbol_text = finger_symbols.get(finger_states_str, 'Unknown')  # Utilise 'Unknown' si le symbole n'est pas dans le dictionnaire


            # Prépare les textes à afficher.
            finger_text = f"Fingers up: {fingers_up}"
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_tip_coords = (int(index_finger_tip.x * 640), int(index_finger_tip.y * 480))
            index_text = f"Index Tip: {index_finger_tip_coords}"
            combined_text = f"{finger_text} - {symbol_text}"

            # Calculate size of the text block for the first line
            combined_text_size, _ = cv2.getTextSize(combined_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            text_width = combined_text_size[0] + 2
            text_height = combined_text_size[1] + 2

            # Calculate size of the text block for the second line
            index_text_size, _ = cv2.getTextSize(index_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

            # Calculate position of the text block
            text_x = x_min
            text_y = y_min - 10 - text_height - index_text_size[1] - 10  # Adjust based on your text placement preference

            # Draw a filled rectangle for text background
            cv2.rectangle(frame, (text_x, text_y), (text_x + text_width, y_min), (0, 0, 0), -1)

            # Display combined text on the first line
            cv2.putText(frame, combined_text, (text_x, text_y + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Display the coordinates of the index finger tip on the second line
            cv2.putText(frame, index_text, (text_x, text_y + text_height + index_text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Show the current frame to the desktop
    cv2.imshow("Frame", frame)
    if cv2.waitKey(5) != -1: # Stop the system
        break
    
cap.release()
cv2.destroyAllWindows()
