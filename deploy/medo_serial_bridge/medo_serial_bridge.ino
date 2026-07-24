/*
 * Medo serial bridge: USB serial -> 12 servos via 2x PCA9685.
 *
 * Replaces the hand-coded trot (see deploy/reference/) with a thin bridge so
 * a host computer can stream joint targets from a trained policy.
 *
 * Hardware: ESP32-S3, PCA9685 at 0x40 (front) + 0x41 (rear),
 *           SDA=21 SCL=20, servos on the channels listed below
 *           (same wiring as the original trot sketch).
 *
 * Protocol (115200 baud, ASCII lines):
 *   "T d0 d1 ... d11\n"  set 12 servo targets in degrees, order:
 *                        fr_hip fr_thigh fr_knee fl_* br_* bl_*
 *                        (hip = abduction, thigh = leg pitch)
 *   "H\n"                go to the home (standing) pose
 *   "P\n"                reply "pong" (link test)
 *
 * Failsafe: if no valid frame arrives for WATCHDOG_MS the bridge eases back
 * to the home pose and stays there until frames resume.
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define PIN_SDA      21
#define PIN_SCL      20
#define PWM_FREQ     50
#define OSC_FREQ     27000000UL
#define PULSE_MIN    150
#define PULSE_MAX    600
#define N_SERVO      12
#define WATCHDOG_MS  400
#define BAUD         115200

Adafruit_PWMServoDriver pcaA(0x40);   // front
Adafruit_PWMServoDriver pcaB(0x41);   // rear

// board (0=A front, 1=B rear) and channel per joint, sim actuator order --
// keep in sync with deploy/servo_map.py
const uint8_t BOARD[N_SERVO] = { 0,0,0, 0,0,0, 1,1,1, 1,1,1 };
const uint8_t CHAN[N_SERVO]  = { 11,4,7, 10,1,2, 14,0,2, 15,5,6 };

// home = standing pose, servo degrees (same table as servo_map.py)
const float HOME_DEG[N_SERVO] = {
  45.0, 76.7, 121.1,    // fr
  35.0, 103.3, 58.9,    // fl
  75.0, 76.7, 121.1,    // br
  45.0, 103.3, 58.9,    // bl
};

float target[N_SERVO];
unsigned long lastFrameMs = 0;
bool inFailsafe = true;

uint16_t deg2pwm(float deg) {
  deg = constrain(deg, 0.0f, 180.0f);
  return (uint16_t)(deg / 180.0f * (PULSE_MAX - PULSE_MIN) + PULSE_MIN);
}

void writeServo(int i, float deg) {
  Adafruit_PWMServoDriver *p = BOARD[i] == 0 ? &pcaA : &pcaB;
  p->setPWM(CHAN[i], 0, deg2pwm(deg));
}

void applyTargets() {
  for (int i = 0; i < N_SERVO; i++) writeServo(i, target[i]);
}

void goHome() {
  for (int i = 0; i < N_SERVO; i++) target[i] = HOME_DEG[i];
  applyTargets();
}

void setup() {
  Serial.begin(BAUD);
  Wire.begin((int8_t)PIN_SDA, (int8_t)PIN_SCL);
  pcaA.begin(); pcaA.setOscillatorFrequency(OSC_FREQ); pcaA.setPWMFreq(PWM_FREQ);
  pcaB.begin(); pcaB.setOscillatorFrequency(OSC_FREQ); pcaB.setPWMFreq(PWM_FREQ);
  delay(100);
  goHome();
  Serial.println("medo bridge ready");
}

void loop() {
  static char buf[160];
  static int  len = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      buf[len] = 0;
      handleLine(buf);
      len = 0;
    } else if (len < (int)sizeof(buf) - 1) {
      buf[len++] = c;
    } else {
      len = 0;  // overlong line: drop it
    }
  }

  if (!inFailsafe && millis() - lastFrameMs > WATCHDOG_MS) {
    inFailsafe = true;
    goHome();
    Serial.println("watchdog: home");
  }
}

void handleLine(char *line) {
  if (line[0] == 'P') { Serial.println("pong"); return; }
  if (line[0] == 'H') { goHome(); inFailsafe = true; Serial.println("home"); return; }
  if (line[0] != 'T') return;

  float vals[N_SERVO];
  char *tok = strtok(line + 1, " ");
  int n = 0;
  while (tok && n < N_SERVO) {
    vals[n++] = atof(tok);
    tok = strtok(NULL, " ");
  }
  if (n != N_SERVO) { Serial.println("err: need 12 values"); return; }

  for (int i = 0; i < N_SERVO; i++) target[i] = vals[i];
  applyTargets();
  lastFrameMs = millis();
  inFailsafe = false;
}
