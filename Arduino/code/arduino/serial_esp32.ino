#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

#define I2C_SDA 8
#define I2C_SCL 9

#define n_samples 3000 //samples to calculate drift every 1ms. Time to compute drift = n_samples / 1000
#define delayTime 100 //must not be too fast, so jetson can handle receivng data

unsigned long lastTime = 0;
float anglePitch = 0;
float angleRoll = 0;
float angleYaw = 0;

float gyroZOffset = 0;

void setup() {
  //2 Serials - 1 windows, 2 jetson
  Serial.begin(115200);
  //Serial0.begin(115200);

  
  Serial.setTimeout(10);
  
  Wire.begin(I2C_SDA, I2C_SCL);
  
  if (!mpu.begin()) {
    while (1) {
      delay(1000);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  Serial.println("Calibrating IMU...");

  for (int i = 0; i < n_samples; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    gyroZOffset += g.gyro.z * 180.0 / PI;
    delay(1);
    //Serial.print(i);
    //Serial.print(", ");
  }
  gyroZOffset /= n_samples;

  lastTime = millis();
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  unsigned long currentTime = millis();
  float dt = (currentTime - lastTime) / 1000.0;
  lastTime = currentTime;

  float accRoll = atan2(a.acceleration.y, a.acceleration.z) * 180.0 / PI;
  float accPitch = atan2(-a.acceleration.x, sqrt(a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z)) * 180.0 / PI;

  float gyroRateX = g.gyro.x * 180.0 / PI;
  float gyroRateY = g.gyro.y * 180.0 / PI;
  float gyroRateZ = g.gyro.z * 180.0 / PI;

  gyroRateZ -= gyroZOffset;

  angleRoll = 0.96 * (angleRoll + gyroRateX * dt) + 0.04 * accRoll;
  anglePitch = 0.96 * (anglePitch + gyroRateY * dt) + 0.04 * accPitch;
  
  angleYaw = angleYaw + gyroRateZ * dt;

  Serial.print("IMU,");
  Serial.print(angleRoll);
  Serial.print(",");
  Serial.print(anglePitch);
  Serial.print(",");
  Serial.println(angleYaw);

  //Serial0.print("IMU,");
  //Serial0.print(angleRoll);
  //Serial0.print(",");
  //Serial0.print(anglePitch);
  //Serial0.print(",");
  //Serial0.println(angleYaw);

  //if (Serial.available() > 0) {
  //  String data = Serial.readStringUntil('\n');
  //  data.trim();
  //  if (data.length() > 0) {
  //    Serial.print("ODEBRANO Z WINDOWS: ");
  //    Serial.println(data);
  //  }
  //}

  //if (Serial0.available() > 0) {
  //  String data = Serial0.readStringUntil('\n');
  //  data.trim();
  //  if (data.length() > 0) {
  //    Serial0.print("ODEBRANO Z JETSONA: ");
  //    Serial0.println(data);
  //  }
  //}

  delay(delayTime);
}