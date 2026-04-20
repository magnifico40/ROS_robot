/**
 * MASTER Controller - ESP32
 * ESP-NOW -> Tank Mix -> 2x UART (Serial1 & Serial2)
 */
#include <esp_now.h>
#include <WiFi.h>

// --- PINY UART DO STEROWNIKÓW ---
#define L_TX_PIN 18  // Połącz z Pinem 12 Slave'a Lewego
#define R_TX_PIN 19  // Połącz z Pinem 12 Slave'a Prawego

// --- STRUKTURA DANYCH RC ---
typedef struct struct_message {
  uint16_t pot1;    // Gaz
  uint16_t pot2;    // Skręt
  uint8_t button1;  
  uint8_t button2;  
} struct_message;

struct_message incomingData;
unsigned long lastRcReceive = 0;

void OnDataRecv(const esp_now_recv_info *info, const uint8_t *data, int len) {
  if (len == sizeof(struct_message)) {
    memcpy(&incomingData, data, sizeof(incomingData));
    lastRcReceive = millis();
  }
}

void setup() {
  Serial.begin(115200); // Debug do PC

  // Inicjalizacja portów UART do Slave'ów
  // Serial1 dla Lewego Silnika
  Serial1.begin(115200, SERIAL_8N1, -1, L_TX_PIN); 
  // Serial2 dla Prawego Silnika
  Serial2.begin(115200, SERIAL_8N1, -1, R_TX_PIN);

  WiFi.mode(WIFI_STA);
  esp_now_init();
  esp_now_register_recv_cb(OnDataRecv);

  Serial.println("Master UART System Ready...");
}

void loop() {
  // Failsafe: 0.5s bez sygnału = STOP
  if (millis() - lastRcReceive > 500) {
    Serial1.println("T0");
    Serial2.println("T0");
    return;
  }

  // Miksowanie (Zakres -5.0 do 5.0 rad/s)
  float throttle = map(incomingData.pot1, 0, 4095, -500, 500) / 100.0f;
  float steering = map(incomingData.pot2, 0, 4095, -300, 300) / 100.0f;

  if (abs(throttle) < 0.15) throttle = 0;
  if (abs(steering) < 0.15) steering = 0;

  float left_v = throttle + steering;
  float right_v = throttle - steering;

  // Wysłanie komend do sterowników
  // zmiana kierunku lewego silnika
  Serial1.print("T"); Serial1.println(-left_v, 2);
  Serial2.print("T"); Serial2.println(right_v, 2);

  delay(20); // Wysyłaj 50 razy na sekundę
}