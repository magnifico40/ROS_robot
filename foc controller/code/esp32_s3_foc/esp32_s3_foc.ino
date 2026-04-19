/**
 * FOC Slave Controller - Autonomous lawn mower
 * ESP32-S3 | Hoverboard BLDC | SimpleFOC
 */

#include <SimpleFOC.h>
#include <Adafruit_NeoPixel.h>

// pins configuration
#define PIN_U 5
#define PIN_V 6
#define PIN_W 7
#define PIN_ENABLE 8

#define PIN_CURRENT_U 1
#define PIN_CURRENT_V 2
#define PIN_CURRENT_W 3

#define PIN_ENC_Z 9
#define PIN_ENC_B 10
#define PIN_ENC_A 11

#define RX1_PIN 12      // RX Mastera
#define TX1_PIN 13      // not used

#define PIN_VBAT 4
#define PIN_WS2812 21
#define NUMPIXELS 1

// motor configuration
#define POLE_PAIRS 15
#define ENCODER_CPR 1024

// battery configuration
#define VBAT_DIV_RATIO ((100.0f + 6.8f) / 6.8f)
#define ADC_REF_VOLTAGE 3.3f
#define ADC_MAX 4095.0f
#define BATTERY_UNDERVOLTAGE 30.0f
#define BATTERY_OVERVOLTAGE  43.0f

// safety configuration
#define MASTER_TIMEOUT_MS 500
#define STALL_CURRENT_THRESHOLD 8.0f
#define STALL_VELOCITY_THRESHOLD 0.5f
#define STALL_TIME_MS 700
#define ENCODER_MAX_VEL_JUMP 80.0f
#define BATTERY_CHECK_INTERVAL 100


BLDCMotor motor = BLDCMotor(POLE_PAIRS);
BLDCDriver3PWM driver = BLDCDriver3PWM(PIN_U, PIN_V, PIN_W, PIN_ENABLE);
Encoder sensor = Encoder(PIN_ENC_A, PIN_ENC_B, ENCODER_CPR, PIN_ENC_Z);
LowsideCurrentSense current_sense = LowsideCurrentSense(0.002f, 20.0f, PIN_CURRENT_U, PIN_CURRENT_V, PIN_CURRENT_W);

Adafruit_NeoPixel strip(NUMPIXELS, PIN_WS2812, NEO_GRB + NEO_KHZ800);

Commander cmdPC = Commander(Serial);
Commander cmdMaster = Commander(Serial1);

// state of motor
bool faultActive = false;
String faultReason = "";
float lastVelocity = 0.0f;
unsigned long stallStart = 0;
unsigned long lastBatCheck = 0;
float filtered_vbat = 42.0f;
unsigned long lastMasterMsg = 0;

// encoder interrupts
void doA() { sensor.handleA(); }
void doB() { sensor.handleB(); }
void doZ() { sensor.handleIndex(); }

// set rgb led
void setLED(uint8_t r, uint8_t g, uint8_t b) {
  strip.setPixelColor(0, strip.Color(r, g, b));
  strip.show();
}

// raise error
void triggerFault(const char* reason) {
  if (!faultActive) {
    faultActive = true;
    faultReason = reason;
    motor.disable();
    Serial.print("FAULT: "); Serial.println(reason);
    setLED(255, 0, 0);
  }
}

float readBatteryVoltage() {
  uint32_t mv = analogReadMilliVolts(PIN_VBAT);
  float pin_voltage = (mv / 1000.0f) * VBAT_DIV_RATIO;
  filtered_vbat = (0.9 * filtered_vbat) + (0.1 * pin_voltage);
  return filtered_vbat;
}

// commands
void doTarget(char* cmd) {
  cmdMaster.scalar(&motor.target, cmd);
  lastMasterMsg = millis();
}

void doMotor(char* cmd)  {
  cmdPC.motor(&motor, cmd);
}

void doReset(char* cmd) {
  if (faultActive) {
    faultActive = false; faultReason = ""; stallStart = 0;
    motor.enable(); setLED(0, 0, 255);
    Serial.println("FAULT RESET: Motor Enabled.");
  }
}

// custom calibration
void customCalibration() {
  Serial.println("\nMOTOR CALIBRATION");

  setLED(255, 0, 255);
  for (float v = 0; v <= motor.voltage_sensor_align; v += 0.1f) {
    motor.setPhaseVoltage(v, 0, 3 * _PI_2);
    delay(15);
  }

  delay(500);
  sensor.update();
  float angle1 = sensor.getAngle();

  for (float v = motor.voltage_sensor_align; v >= 0; v -= 0.1f) {
    motor.setPhaseVoltage(v, 0, 3 * _PI_2);
    delay(5);
  }

  // dir
  for (float v = 0; v <= motor.voltage_sensor_align; v += 0.1f) {
    motor.setPhaseVoltage(v, 0, 3 * _PI_2 + _PI_6);
    delay(15);
  }

  delay(500);
  sensor.update();
  float angle2 = sensor.getAngle();

  for (float v = motor.voltage_sensor_align; v >= 0; v -= 0.1f) {
    motor.setPhaseVoltage(v, 0, 3 * _PI_2 + _PI_6);
    delay(5);
  }

  if (angle2 > angle1){
    motor.sensor_direction = Direction::CW;
  }
  else{
    motor.sensor_direction = Direction::CCW;
  }
  motor.setPhaseVoltage(0, 0, 0);
  delay(500);

  // searching Z impulse
  setLED(255, 255, 0);
  float angle_openloop = sensor.getAngle();
  while(sensor.needsSearch()) {
    angle_openloop += motor.velocity_index_search * 0.001f;
    motor.setPhaseVoltage(motor.voltage_sensor_align * 0.5f, 0, _electricalAngle(angle_openloop, motor.pole_pairs));
    sensor.update();
    delay(1);
  }

  motor.setPhaseVoltage(0, 0, 0);
  delay(500);

  // 4. Finalna synchronizacja
  for (float v = 0; v <= motor.voltage_sensor_align; v += 0.1f) {
    motor.setPhaseVoltage(v, 0, 3 * _PI_2);
    delay(15);
  }

  delay(500);
  sensor.update();
  motor.zero_electric_angle = _electricalAngle(sensor.getAngle(), motor.pole_pairs);

  for (float v = motor.voltage_sensor_align; v >= 0; v -= 0.1f) {
    motor.setPhaseVoltage(v, 0, 3 * _PI_2);
    delay(5);
  }
  Serial.println("CALIBRATION DONE");
}

// setup
void setup() {
  Serial.begin(115200);
  Serial1.begin(115200, SERIAL_8N1, RX1_PIN, TX1_PIN);
  analogReadResolution(12);

  strip.begin();
  strip.setBrightness(20);
  setLED(255, 100, 0);

  sensor.pullup = Pullup::USE_INTERN;
  sensor.init();
  sensor.enableInterrupts(doA, doB, doZ);
  motor.linkSensor(&sensor);

  driver.voltage_power_supply = 42.0f;
  driver.init();
  motor.linkDriver(&driver);

  current_sense.linkDriver(&driver);
  current_sense.init();
  motor.linkCurrentSense(&current_sense);

  motor.controller = MotionControlType::velocity;
  motor.voltage_limit = 15.0f;
  motor.velocity_limit = 5.0f;
  motor.current_limit = 10.0f;

  motor.voltage_sensor_align = 4.0f;
  motor.velocity_index_search = 1.0f;

  // PI controller parameters
  motor.PID_velocity.P = 1.85f;
  motor.PID_velocity.I = 0.6f;
  // low pass filter
  motor.LPF_velocity.Tf = 0.05f;
  // max current change per second
  motor.PID_velocity.output_ramp = 80;

  motor.init();

  // check if battery connected
  while (readBatteryVoltage() < BATTERY_UNDERVOLTAGE) {
    setLED(255, 100, 0); delay(250); setLED(0, 0, 0); delay(250);
  }

  customCalibration();
  motor.initFOC();

  // Serial commands
  cmdMaster.add('T', doTarget, "Target");
  cmdPC.add('M', doMotor, "Motor tune");
  cmdPC.add('R', doReset, "Reset Fault");

  setLED(0, 0, 255);
  lastMasterMsg = millis();
}

// loop
void loop() {
  cmdPC.run();
  cmdMaster.run();

  if (!faultActive) {
    motor.loopFOC();

    // FAILSAFE: no data from Master
    if (millis() - lastMasterMsg > MASTER_TIMEOUT_MS) {
      motor.target = 0;
    }

    static unsigned long undervoltage_timer = 0;
    // check battery voltage
    if (millis() - lastBatCheck > BATTERY_CHECK_INTERVAL) {
      lastBatCheck = millis();
      float vbat = readBatteryVoltage();

      if (vbat > BATTERY_OVERVOLTAGE) triggerFault("Battery - overvoltage");
      if (vbat < BATTERY_UNDERVOLTAGE) {
        if (undervoltage_timer == 0) undervoltage_timer = millis();
        if (millis() - undervoltage_timer > 3000) {
          triggerFault("Battery - Undervoltage");
        }
      } else {
        undervoltage_timer = 0;
      }

    }

    // encoder jump detection
    float vel = sensor.getVelocity();
    if (fabs(vel - lastVelocity) > ENCODER_MAX_VEL_JUMP) triggerFault("Encoder error");
    lastVelocity = vel;

    // stall detection
    if (fabs(motor.current.q) > STALL_CURRENT_THRESHOLD && fabs(vel) < STALL_VELOCITY_THRESHOLD && fabs(motor.target) > 0.5f) {
      if (stallStart == 0) stallStart = millis();
      if (millis() - stallStart > STALL_TIME_MS) triggerFault("obstacle - motor blocked");
    } else { stallStart = 0; }

    motor.move();
  }
}