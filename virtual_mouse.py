import cv2
import mediapipe as mp
import pyautogui
import numpy as np

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Virtual Mouse", cv2.WND_PROP_TOPMOST, 1)

while True:
    success, frame = cap.read()
    if not success or frame is None:
        continue 

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y)

            thumb_tip = lm[4]
            index_tip = lm[8]
            distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            if distance < 0.03:
                pyautogui.click()
                pyautogui.sleep(0.2)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
