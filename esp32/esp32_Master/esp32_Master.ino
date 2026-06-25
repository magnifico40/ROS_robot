#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <Adafruit_BNO08x.h>
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

#define BLDC_PIN 21
#define ESC_STOP 900
#define ESC_RUN 1300
#define ESC_PWM_FREQ 50
#define ESC_PWM_RES 14

Adafruit_BNO08x bno08x;
sh2_SensorValue_t sensorValue;

float imu_ax = 0, imu_ay = 0, imu_az = 0;
float imu_gx = 0, imu_gy = 0, imu_gz = 0;
float imu_qr = 1.0, imu_qi = 0, imu_qj = 0, imu_qk = 0;

// ZMIENNE ODOMETRII Z KÓŁ
float odom_L_angle = 0.0f, odom_L_vel = 0.0f;
float odom_R_angle = 0.0f, odom_R_vel = 0.0f;

HardwareSerial SerialGPS(2);

typedef struct struct_message {
    uint16_t pot1;
    uint16_t pot2;
    uint8_t button1;
    uint8_t button2;
} struct_message;

struct_message incomingData;
unsigned long lastRcReceive = 0;

// Bufory odczytu
String inputString = "";
String gpsBuffer = "";
String lMotorBuffer = "";
String rMotorBuffer = "";
bool lastButton1 = 0;

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

void writeESC(int microseconds) {
    uint32_t duty = (microseconds * 16383) / 20000;
    ledcWrite(BLDC_PIN, duty);
}

void setup() {
    Serial.setRxBufferSize(4096);
    Serial.begin(230400);
    
    Serial0.begin(115200, SERIAL_8N1, L_MOTOR_RX, L_MOTOR_TX);
    Serial1.begin(115200, SERIAL_8N1, R_MOTOR_RX, R_MOTOR_TX);

    SerialGPS.setRxBufferSize(2048);
    SerialGPS.setTxBufferSize(1024);
    SerialGPS.begin(115200, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

    // Ochrona pamięci przed fragmentacją przy ciągłym dodawaniu znaków
    inputString.reserve(256);
    gpsBuffer.reserve(256);
    lMotorBuffer.reserve(64);
    rMotorBuffer.reserve(64);

    // Szybka magistrala I2C
    Wire.begin(IMU_SDA, IMU_SCL);
    Wire.setClock(400000); 

    if (!bno08x.begin_I2C(0x4A, &Wire) && !bno08x.begin_I2C(0x4B, &Wire)) {
        Serial.println("Error: BNO085 not found");
    } else {
        bno08x.enableReport(SH2_ACCELEROMETER, 100000);
        bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 100000);
        bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 100000);
        Serial.println("BNO085 initialized");
    }

    WiFi.mode(WIFI_STA);
    if (esp_now_init() == ESP_OK) {
        esp_now_register_recv_cb((esp_now_recv_cb_t)OnDataRecv);
    }

    ledcAttach(BLDC_PIN, ESC_PWM_FREQ, ESC_PWM_RES);
    writeESC(ESC_STOP);
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

// Parsowanie odometri z silników Format: "O kat predkosc"
void parseLeftMotorOdom(String &cmd) {
    if (cmd.startsWith("O ")) {
        int spaceIdx = cmd.indexOf(' ', 2);
        if (spaceIdx > 0) {
            odom_L_angle = cmd.substring(2, spaceIdx).toFloat();
            odom_L_vel = cmd.substring(spaceIdx + 1).toFloat();
        }
    }
}

void parseRightMotorOdom(String &cmd) {
    if (cmd.startsWith("O ")) {
        int spaceIdx = cmd.indexOf(' ', 2);
        if (spaceIdx > 0) {
            odom_R_angle = cmd.substring(2, spaceIdx).toFloat();
            odom_R_vel = cmd.substring(spaceIdx + 1).toFloat();
        }
    }
}

void processRosCommand(String &cmd) {
    if (cmd.startsWith("M,") && incomingData.button1 == 1) {
        int firstComma = cmd.indexOf(',');
        int secondComma = cmd.indexOf(',', firstComma + 1);
        if (firstComma > 0 && secondComma > firstComma) {
            float vL = cmd.substring(firstComma + 1, secondComma).toFloat();
            float vR = cmd.substring(secondComma + 1).toFloat();
            sendToMotors(vL, vR);
        }
    }
    else if (cmd.startsWith("RTCM,")) {
        cmd.trim();
        const char* hexStr = cmd.c_str() + 5; 
        size_t hexLen = strlen(hexStr);
        size_t dataLen = hexLen / 2;
        
        if (dataLen > 0 && dataLen < 2000) {
            uint8_t binBuffer[2000]; 
            for (size_t i = 0; i < dataLen; i++) {
                binBuffer[i] = (hexToByte(hexStr[i * 2]) << 4) | hexToByte(hexStr[i * 2 + 1]);
            }
            SerialGPS.write(binBuffer, dataLen);
        }
    }
    else if (cmd.startsWith("BLDC,") && incomingData.button1 == 1) {
        int status = cmd.substring(5).toInt();
        if (status == 1) {
            writeESC(ESC_RUN);
        } else {
            writeESC(ESC_STOP);
        }
    }
}

void loop() {
    // ASYNCHRONICZNY ODCZYT ROS
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            processRosCommand(inputString);
            inputString = "";
        } else {
            inputString += c;
        }
    }

    // ASYNCHRONICZNY ODCZYT GPS
    while (SerialGPS.available()) {
        char c = SerialGPS.read();
        gpsBuffer += c;
        if (c == '\n') {
            Serial.print(gpsBuffer); 
            gpsBuffer = ""; 
        }
    }

    // ASYNCHRONICZNY ODCZYT ODOMETRII - LEWE KOŁO
    while (Serial0.available()) {
        char c = Serial0.read();
        if (c == '\n') {
            parseLeftMotorOdom(lMotorBuffer);
            lMotorBuffer = "";
        } else {
            lMotorBuffer += c;
        }
    }

    // ASYNCHRONICZNY ODCZYT ODOMETRII - PRAWE KOŁO
    while (Serial1.available()) {
        char c = Serial1.read();
        if (c == '\n') {
            parseRightMotorOdom(rMotorBuffer);
            rMotorBuffer = "";
        } else {
            rMotorBuffer += c;
        }
    }

    // OBSŁUGA RESTARTU IMU
    if (bno08x.wasReset()) {
        bno08x.enableReport(SH2_ACCELEROMETER, 100000);
        bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 100000);
        bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 100000);
    }

    // ZDEJMOWANIE CAŁEJ KOLEJKI Z BNO085
    while (bno08x.getSensorEvent(&sensorValue)) {
        switch (sensorValue.sensorId) {
            case SH2_ACCELEROMETER:
                imu_ax = sensorValue.un.accelerometer.x;
                imu_ay = sensorValue.un.accelerometer.y;
                imu_az = sensorValue.un.accelerometer.z;
                break;
            case SH2_GYROSCOPE_CALIBRATED:
                imu_gx = sensorValue.un.gyroscope.x;
                imu_gy = sensorValue.un.gyroscope.y;
                imu_gz = sensorValue.un.gyroscope.z;
                break;
            case SH2_GAME_ROTATION_VECTOR:
                imu_qr = sensorValue.un.gameRotationVector.real;
                imu_qi = sensorValue.un.gameRotationVector.i;
                imu_qj = sensorValue.un.gameRotationVector.j;
                imu_qk = sensorValue.un.gameRotationVector.k;
                break;
        }
    }

    // PUBLIKOWANIE DANYCH IMU DO ROS
    static unsigned long lastImu = 0;
    if (millis() - lastImu >= 100) {
        lastImu = millis();
        Serial.printf("IMU,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
            imu_ax, imu_ay, imu_az,
            imu_gx, imu_gy, imu_gz,
            imu_qr, imu_qi, imu_qj, imu_qk);
    }

    // PUBLIKOWANIE ODOMETRII DO ROS - 20Hz
    static unsigned long lastOdomOut = 0;
    if (millis() - lastOdomOut >= 50) {
        lastOdomOut = millis();
        Serial.printf("ODOM,%.3f,%.3f,%.3f,%.3f\n", 
                      odom_L_angle, odom_L_vel, 
                      odom_R_angle, odom_R_vel);
    }

    // BEZPIECZEŃSTWO I STEROWANIE MANUALNE
    static unsigned long lastMotorUpdate = 0;
    if (millis() - lastMotorUpdate > 50) {
        lastMotorUpdate = millis();

        if (millis() - lastRcReceive > 500) {
            sendToMotors(0, 0);
            writeESC(ESC_STOP);
        }
        else if (incomingData.button1 == 0) {
            float throttle = map(incomingData.pot1, 0, 4095, (int)(-MAX_SPEED*100), (int)(MAX_SPEED*100)) / 100.0f;
            float steering = map(incomingData.pot2, 0, 4095, (int)(-MAX_SPEED*50), (int)(MAX_SPEED*50)) / 100.0f; 

            if (abs(throttle) < 0.15) throttle = 0;
            if (abs(steering) < 0.15) steering = 0;

            sendToMotors(throttle + steering, throttle - steering);

            if(incomingData.button2 == 1){
                writeESC(ESC_RUN);
            }else{
                writeESC(ESC_STOP);
            }
        }
        else if (incomingData.button1 == 1 && lastButton1 == 0){
            writeESC(ESC_STOP);
        }
        lastButton1 = incomingData.button1;
    }
}