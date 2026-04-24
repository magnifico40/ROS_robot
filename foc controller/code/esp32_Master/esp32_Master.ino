#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <Adafruit_MPU6050.h>
#include <Wire.h>


#define MAX_SPEED 2.5f      // m/s
#define WHEEL_RADIUS 0.099f // 19,8 cm / 2
#define L_MOTOR_RX 15
#define L_MOTOR_TX 16
#define R_MOTOR_RX 17
#define R_MOTOR_TX 18
#define GPS_RX_PIN 4
#define GPS_TX_PIN 5
#define IMU_SDA 8
#define IMU_SCL 9

Adafruit_MPU6050 mpu;
HardwareSerial SerialGPS(2);

typedef struct struct_message {
    uint16_t pot1;
    uint16_t pot2;
    uint8_t button1;
    uint8_t button2;
} struct_message;

struct_message incomingData;
unsigned long lastRcReceive = 0;
String inputString = "";
String gpsBuffer = "";

void OnDataRecv(const esp_now_recv_info *info, const uint8_t *data, int len) {
    if (len == sizeof(struct_message)) {
        memcpy(&incomingData, data, sizeof(incomingData));
        lastRcReceive = millis();
    }
}

uint8_t hexToByte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

void setup() {
    Serial.begin(115200);

    Serial0.begin(115200, SERIAL_8N1, L_MOTOR_RX, L_MOTOR_TX);
    Serial1.begin(115200, SERIAL_8N1, R_MOTOR_RX, R_MOTOR_TX);

    SerialGPS.begin(115200, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

    Wire.begin(IMU_SDA, IMU_SCL);
    mpu.begin();

    WiFi.mode(WIFI_STA);
    if (esp_now_init() == ESP_OK) {
        esp_now_register_recv_cb((esp_now_recv_cb_t)OnDataRecv);
    }
}

void sendToMotors(float vL_ms, float vR_ms) {
    vL_ms = constrain(vL_ms, -MAX_SPEED, MAX_SPEED);
    vR_ms = constrain(vR_ms, -MAX_SPEED, MAX_SPEED);

    float omegaL = vL_ms / WHEEL_RADIUS;
    float omegaR = vR_ms / WHEEL_RADIUS;

    Serial0.print("T");
    Serial0.println(-omegaL, 2);
    Serial1.print("T");
    Serial1.println(omegaR, 2);
}

void processRosCommand(String cmd) {
    if (cmd.startsWith("M,") && incomingData.button1 == 1) {
        int firstComma = cmd.indexOf(',');
        int secondComma = cmd.indexOf(',', firstComma + 1);
        float vL = cmd.substring(firstComma + 1, secondComma).toFloat();
        float vR = cmd.substring(secondComma + 1).toFloat();

        sendToMotors(vL, vR);
    }
    else if (cmd.startsWith("RTCM,")) {
        String hex = cmd.substring(5);
        for (int i = 0; i < hex.length(); i += 2) {
            uint8_t b = (hexToByte(hex[i]) << 4) | hexToByte(hex[i + 1]);
            SerialGPS.write(b);
        }
    }
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            processRosCommand(inputString);
            inputString = "";
        } else {
            inputString += c;
        }
    }

    // send gps data
    while (SerialGPS.available()) {
        char c = SerialGPS.read();
        gpsBuffer += c;
        if (c == '\n') {
            Serial.print(gpsBuffer);
            gpsBuffer = "";
        }
    }

    // send imu data
    static unsigned long lastImu = 0;
    if (millis() - lastImu > 20) {
        lastImu = millis();
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        Serial.printf("IMU,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
            a.acceleration.x, a.acceleration.y, a.acceleration.z,
            g.gyro.x, g.gyro.y, g.gyro.z);
    }

    // steering mode (manual/autonomous)
    static unsigned long lastMotorUpdate = 0;
    if (millis() - lastMotorUpdate > 50) {
        lastMotorUpdate = millis();

        if (millis() - lastRcReceive > 500) {
            sendToMotors(0, 0);
        }
        else if (incomingData.button1 == 0) {
            float throttle = map(incomingData.pot1, 0, 4095, (int)(-MAX_SPEED*100), (int)(MAX_SPEED*100)) / 100.0f;
            float steering = map(incomingData.pot2, 0, 4095, (int)(-MAX_SPEED*50), (int)(MAX_SPEED*50)) / 100.0f; 

            if (abs(throttle) < 0.15) throttle = 0;
            if (abs(steering) < 0.15) steering = 0;

            sendToMotors(throttle + steering, throttle - steering);
        }
    }
}