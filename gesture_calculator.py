import cv2
import mediapipe as mp
import numpy as np
import time
import math

class Button:
    def __init__(self, x, y, w, h, label):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label

    def is_inside(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, img, theme, hovered=False, active=False):
        # Draw transparent filled rectangle with alpha
        overlay = img.copy()
        alpha_fill = 0.3
        color_fill = theme['btn_active'] if active else (theme['btn_hover'] if hovered else theme['btn'])
        cv2.rectangle(overlay, (self.x, self.y), (self.x + self.w, self.y + self.h), color_fill, -1)
        cv2.addWeighted(overlay, alpha_fill, img, 1 - alpha_fill, 0, img)

        # Draw button border (opaque)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), theme['border'], 2)

        # Draw text (opaque)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        txt_size = cv2.getTextSize(self.label, font, scale, thickness)[0]
        tx = self.x + (self.w - txt_size[0]) // 2
        ty = self.y + (self.h + txt_size[1]) // 2
        cv2.putText(img, self.label, (tx, ty), font, scale, theme['text'], thickness, cv2.LINE_AA)

THEMES = [
    {'btn': (0, 255, 255), 'btn_hover': (0, 180, 180), 'btn_active': (0, 120, 120),
     'text': (255, 255, 255), 'border': (255, 255, 255)},
    {'btn': (255, 255, 255), 'btn_hover': (200, 200, 200), 'btn_active': (150, 150, 150),
     'text': (0, 0, 0), 'border': (0, 0, 0)},
]

LAYOUT = [
    ["C", "(", ")", "/"],
    ["7", "8", "9", "*"],
    ["4", "5", "6", "-"],
    ["1", "2", "3", "+"],
    ["0", ".", "=", "T"]
]

def safe_eval(expr):
    allowed = set("0123456789.+-*/() ")
    if not expr:
        return ""
    if any(ch not in allowed for ch in expr):
        return "ERR"
    try:
        result = eval(expr)
        return str(result)
    except Exception:
        return "ERR"

def count_fingers(hand_landmarks, handedness_label=None):
    lm = hand_landmarks.landmark
    if handedness_label is None:
        thumb_is_open = lm[4].x < lm[3].x
    else:
        if handedness_label == "Right":
            thumb_is_open = lm[4].x < lm[3].x
        else:
            thumb_is_open = lm[4].x > lm[3].x
    fingers = [1 if thumb_is_open else 0]
    tips_pips = [(8,6), (12,10), (16,14), (20,18)]
    for tip, pip in tips_pips:
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)
    return sum(fingers), fingers

def create_buttons(frame_w, frame_h, display_h):
    btn_w = frame_w // 4
    btn_h = (frame_h - display_h) // 5
    buttons = []
    for i, row in enumerate(LAYOUT):
        for j, label in enumerate(row):
            x = j * btn_w
            y = display_h + i * btn_h
            buttons.append(Button(x, y, btn_w, btn_h, label))
    return buttons

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    expression = ""
    theme_idx = 0
    theme = THEMES[theme_idx]

    prev_click = 0
    click_cooldown = 0.45
    last_theme_switch = 0
    theme_cooldown = 1.0

    ret, frame = cap.read()
    if not ret:
        print("No camera")
        return
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    half_w = w // 2
    display_h = int(h * 0.15)

    buttons = create_buttons(half_w, h, display_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Left half: stable face crop
        face_crop = frame[:, half_w//4 : half_w + half_w//4]
        face_resized = cv2.resize(face_crop, (half_w, h))

        # Right half: gesture background + transparent buttons/text
        calc_bg = frame[:, half_w:].copy()
        img_rgb = cv2.cvtColor(calc_bg, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        pointer = None
        pinch = False
        five_fingers = False
        handedness_label = None

        if results.multi_hand_landmarks:
            if results.multi_handedness:
                handedness_label = results.multi_handedness[0].classification[0].label
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(calc_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            ix, iy = int(lm[8].x * half_w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * half_w), int(lm[4].y * h)
            pointer = (ix, iy)
            dist_norm = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
            pinch = dist_norm < 0.045
            cnt, _ = count_fingers(hand_landmarks, handedness_label)
            five_fingers = (cnt == 5)
            cv2.circle(calc_bg, pointer, 10, (0, 255, 255), -1)

        # No background rectangles here — fully transparent!

        # Draw expression text on top-left (slightly shadowed for visibility)
        cv2.putText(calc_bg, expression, (10, display_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, theme['text'], 3, cv2.LINE_AA)

        # Draw buttons with transparent fills
        for btn in buttons:
            hovered = False
            active = False
            if pointer and btn.is_inside(pointer[0], pointer[1]):
                hovered = True
                if pinch and (time.time() - prev_click) > click_cooldown and not five_fingers:
                    prev_click = time.time()
                    active = True
                    label = btn.label
                    if label == "C":
                        expression = ""
                    elif label == "=":
                        expression = safe_eval(expression)
                    elif label == "T":
                        theme_idx = (theme_idx + 1) % len(THEMES)
                        theme = THEMES[theme_idx]
                    else:
                        expression += label
            btn.draw(calc_bg, theme, hovered=hovered, active=active)

        # Theme switch with 5 fingers
        if five_fingers and (time.time() - last_theme_switch) > theme_cooldown:
            last_theme_switch = time.time()
            theme_idx = (theme_idx + 1) % len(THEMES)
            theme = THEMES[theme_idx]
            cv2.putText(calc_bg, "Theme switched", (10, display_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, theme['text'], 2)

        combined = np.hstack((face_resized, calc_bg))

        cv2.imshow("Face + Transparent Gesture Calculator", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
