#!/usr/bin/env python3

import cv2
import time

# ==========================================
# SEKCJA KONFIGURACJI - ZMIENIAJ TYLKO TU
# ==========================================

# 1. Rozdzielczość SENSORA (Co pobieramy z kamery)
# 3264x2464 (21 fps) - Pełna matryca
# 1640x1232 (30 fps) - Pełne pole widzenia (Binning 2x2) <--- ZALECANE DO FUZJI
# 1280x720  (60 fps) - Wąskie pole widzenia (Crop 16:9)
SENSOR_WIDTH = 1640
SENSOR_HEIGHT = 1232
FRAMERATE = 30
#DISPLAY_WIDTH = 1640
#DISPLAY_HEIGHT = 1232
# 2. Rozdzielczość WYNIKOWA (Do jakiej skalujemy)
# Tu wpisz 640x480, aby sprawdzić czy obraz jest skalowany czy ucinany.
# Jeśli wpiszesz tu wartości mniejsze niż SENSOR, Jetson użyje sprzętowego skalera (VIC).
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# 3. Obrót (0 = brak, 2 = 180 stopni)
FLIP_METHOD = 2 

# ==========================================

def get_gstreamer_pipeline():
    """
    Generuje string konfiguracyjny dla GStreamera na podstawie zmiennych powyżej.
    """
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink"
        % (
            SENSOR_WIDTH, SENSOR_HEIGHT, FRAMERATE,
            FLIP_METHOD,
            DISPLAY_WIDTH, DISPLAY_HEIGHT
        )
    )

def main():
    pipeline = get_gstreamer_pipeline()
    print(f"Uruchamianie z pipeline:\n{pipeline}\n")
    print("-" * 50)
    print(f"SENSOR:  {SENSOR_WIDTH}x{SENSOR_HEIGHT} @ {FRAMERATE}fps")
    print(f"WYJŚCIE: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print("-" * 50)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("BŁĄD: Nie można otworzyć kamery. Sprawdź:")
        print("1. Czy inna aplikacja (ROS?) nie blokuje kamery.")
        print("2. Czy zrestartowałeś kontener po restarcie nvargus-daemon.")
        return

    print("Kamera otwarta. Naciśnij 'q', aby zakończyć.")
    
    # Licznik FPS do weryfikacji wydajności
    prev_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Błąd odczytu klatki.")
                break

            # Obliczanie rzeczywistego FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # Wypisanie info na obrazie
            info_text = f"In: {SENSOR_WIDTH}x{SENSOR_HEIGHT} Out: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} FPS: {fps:.1f}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Wyświetlanie
            cv2.imshow("Test Rozdzielczosci", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Zakończono.")

if __name__ == "__main__":
    main()