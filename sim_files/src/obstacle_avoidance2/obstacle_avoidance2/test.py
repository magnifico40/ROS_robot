import fields2cover as f2c

# Przeszukujemy całą bibliotekę f2c pod kątem słowa "Trapez" lub "Decomp"
for nazwa in dir(f2c):
    if "Trapez" in nazwa or "Decomp" in nazwa:
        print(nazwa)