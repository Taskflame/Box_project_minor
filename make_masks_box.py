import os, glob, cv2
import numpy as np

# === Папки ===
FRAMES_DIR = "frames"
MASKS_DIR = "masks"
os.makedirs(MASKS_DIR, exist_ok=True)

# === Цветовые границы для синей коробки ===
LOWER_BLUE = np.array([80, 40, 40])
UPPER_BLUE = np.array([135, 255, 255])

for i, path in enumerate(sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))):
    frame = cv2.imread(path)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # === Маска по синему цвету ===
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # === Заполнение внутренних дырок ===
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    mask = filled_mask

    # === Морфологические операции для сглаживания ===
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # === Повторное заполнение, включая вложенные контуры ===
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for j, cnt in enumerate(contours):
            if hierarchy[0][j][3] == -1:  # внешний контур
                cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

    # === Немного размыть границы — сгладит “ступеньки” ===
    mask = cv2.medianBlur(mask, 15)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # === Сохранение результата ===
    out_path = os.path.join(MASKS_DIR, os.path.basename(path))
    cv2.imwrite(out_path, mask)
    print(f"[{i:03d}] saved {out_path}")

print("✅ Маски коробки пересчитаны с расширением и заливкой контуров.")
