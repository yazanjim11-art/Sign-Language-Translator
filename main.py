import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_sign_label(fingers, lms):
    thumb_open = lms[4].x < lms[3].x 
    dist_all_to_thumb = sum([calculate_distance(lms[4], lms[i]) for i in [8, 12, 16, 20]])

    if dist_all_to_thumb < 0.2:
        return "Eat"
    if thumb_open and sum(fingers) == 0 and lms[4].y < lms[3].y:
        return "Drink"
    if fingers == [1, 0, 0, 0] and lms[8].y > lms[6].y:
        return "Me"
    if fingers == [1, 0, 0, 0] and lms[8].y < lms[6].y:
        return "You"
    if sum(fingers) == 4 and thumb_open:
        return "Stop"
    if dist_all_to_thumb < 0.15:
        return "Zero"
    if thumb_open and fingers == [1, 0, 0, 1]:
        return "I Love You"
    if calculate_distance(lms[4], lms[8]) < 0.05 and fingers[1:] == [1, 1, 1]:
        return "Great"
    if fingers == [1, 1, 0, 0] and calculate_distance(lms[8], lms[12]) >= 0.04:
        return "Peace"
    if fingers == [1, 1, 0, 0] and calculate_distance(lms[8], lms[12]) < 0.04:
        return "No"
    if sum(fingers) == 0 and not thumb_open:
        return "Yes"
    if thumb_open and sum(fingers) == 0:
        return "Help"
    if thumb_open and fingers == [0, 0, 0, 1]:
        return "How are you"

    return "Analyzing..."

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Yazan AI - Sign Language Translator is running successfully!")

while cap.isOpened():
    success, img = cap.read()
    if not success: 
        break
    
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

            fingers = []
            for tip_id in [8, 12, 16, 20]:
                if hand_lms.landmark[tip_id].y < hand_lms.landmark[tip_id - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            sign_text = get_sign_label(fingers, hand_lms.landmark)

            box_start_x = w // 2 - 250
            box_end_x = w // 2 + 250
            box_start_y = h - 120
            box_end_y = h - 30

            cv2.rectangle(img, (box_start_x, box_start_y), (box_end_x, box_end_y), (30, 30, 30), -1)
            cv2.rectangle(img, (box_start_x, box_start_y), (box_end_x, box_end_y), (0, 255, 0), 2)
            cv2.putText(img, sign_text, (box_start_x + 20, box_start_y + 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    cv2.imshow("Sign Language System - Final Pro", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()