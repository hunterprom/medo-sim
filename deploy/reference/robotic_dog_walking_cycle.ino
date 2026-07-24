 /*
 * Robotic Dog – Forward Trot Gait
 * ─────────────────────────────────────────────────────────────────────────────
 * Hardware : ESP32-S3  |  2x PCA9685  (SDA=21, SCL=20)
 * Leg IK   : 2-DOF planar  –  hip = 80 mm, knee = 60 mm
 * Gait     : trot "01/10"  –  diagonal pairs alternate swing <-> stance
 *
 * Install via Arduino Library Manager:
 *   "Adafruit PWM Servo Driver Library" by Adafruit
 *
 * Servo wiring
 * ─────────────────────────────────────────────────────────────────────────────
 *   PCA-A (0x40) – front legs:
 *     ch 1 = FL hip      ch 4 = FR hip
 *     ch 2 = FL knee     ch 7 = FR knee
 *
 *   PCA-B (0x41) – rear legs:
 *     ch 5 = RL hip      ch 0 = RR hip
 *     ch 6 = RL knee     ch 2 = RR knee
 *
 * CALIBRATION
 * ─────────────────────────────────────────────────────────────────────────────
 *  1. Upload – robot stands still for 1.5 s.
 *  2. Adjust hip_trim / knee_trim until all legs are at equal height.
 *  3. If a hip swings the wrong direction, flip hip_dir  (+1 <-> -1).
 *  4. If a knee bends the wrong way,       flip knee_dir (+1 <-> -1).
 *  5. Tune STANCE_H, STEP_LEN, STEP_H, SPEED_HZ for your build.
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ── I2C & PCA9685 ─────────────────────────────────────────────────────────────
#define PIN_SDA      21
#define PIN_SCL      20
#define ADDR_PCA_A   0x40
#define ADDR_PCA_B   0x41

#define PWM_FREQ     50
#define OSC_FREQ     27000000UL
#define PULSE_MIN    150
#define PULSE_MAX    600

// ── Leg geometry (mm) ─────────────────────────────────────────────────────────
const float L_HIP  = 80.0;   // hip joint -> knee joint
const float L_KNEE = 60.0;   // knee joint -> foot
const float L_MAX  = L_HIP + L_KNEE;

// ── Gait parameters ───────────────────────────────────────────────────────────
// STANCE_H - STEP_H must stay above 112 mm to keep knees within servo range
const float STANCE_H    = 135.0;  // standing height (mm)
const float STEP_LEN    =  60.0;  // stride length (mm)
const float STEP_H      =  22.0;  // swing lift height (mm)
const float STANCE_LIFT =   6.0;  // stance push-up height (mm)
const float SPEED_HZ    =   1.0;  // half-cycles per second

// ── Hip servo fixed angles (degrees) – change to your liking ─────────────────
const float HIP_FL = 35.0;   // board A ch10
const float HIP_FR = 45.0;   // board A ch11
const float HIP_RL = 45.0;   // board B ch15
const float HIP_RR = 75.0;   // board B ch14

// ── Leg descriptors ───────────────────────────────────────────────────────────
struct LegDef {
  Adafruit_PWMServoDriver *pca;
  uint8_t hip_ch, knee_ch;
  float   hip_dir;
  float   knee_dir;
  float   hip_trim;
  float   knee_trim;
};

Adafruit_PWMServoDriver pcaA(ADDR_PCA_A);
Adafruit_PWMServoDriver pcaB(ADDR_PCA_B);

LegDef legs[4] = {
//          pca    hip  kn  hdir   kdir  htrim  ktrim
  { &pcaA,   1,   2, -1.0, +1.0, 90.0, 90.0 }, // 0  FL
  { &pcaA,   4,   7, +1.0, -1.0, 90.0, 90.0 }, // 1  FR
  { &pcaB,   5,   6, -1.0, +1.0, 90.0, 90.0 }, // 2  RL
  { &pcaB,   0,   2, +1.0, -1.0, 90.0, 90.0 }, // 3  RR
};

// ── Gait state ────────────────────────────────────────────────────────────────
#define GAIT_STEPS 10

int   gPhase = 0;
int   gStep  = 0;          // current step index 0..9

// Precomputed 10-point semicircle for swing and stance
float swingX[GAIT_STEPS];
float swingY[GAIT_STEPS];
float stanceX[GAIT_STEPS];
float stanceY[GAIT_STEPS];

void precomputeGait() {
  for (int i = 0; i < GAIT_STEPS; i++) {
    float angle = PI * i / (GAIT_STEPS - 1);   // 0 .. PI in 10 steps

    // Semicircle: x uses cosine, y uses sine  (true half-circle arc)
    swingX[i] = -(STEP_LEN / 2.0) * cos(angle);          // rear -> front
    swingY[i] =  STANCE_H - STEP_H * sin(angle);          // lifted in middle

    // Stance: foot slides back with a small upward arc in the middle (push-off)
    stanceX[i] = (STEP_LEN / 2.0) - STEP_LEN * i / (GAIT_STEPS - 1);
    stanceY[i] =  STANCE_H - STANCE_LIFT * sin(PI * i / (GAIT_STEPS - 1));
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
uint16_t deg2pwm(float deg) {
  deg = constrain(deg, 0.0, 180.0);
  return (uint16_t)(deg / 180.0 * (PULSE_MAX - PULSE_MIN) + PULSE_MIN);
}

void writeServo(Adafruit_PWMServoDriver *pca, uint8_t ch, float deg) {
  pca->setPWM(ch, 0, deg2pwm(deg));
}

// ── 2-DOF Planar Inverse Kinematics ──────────────────────────────────────────
bool ikSolve(float x, float y, float &hipDeg, float &kneeDeg) {
  float d2 = x*x + y*y;
  float d  = sqrt(d2);
  if (d > L_MAX || d < fabs(L_HIP - L_KNEE)) return false;

  float cosK = (d2 - L_HIP*L_HIP - L_KNEE*L_KNEE) / (2.0 * L_HIP * L_KNEE);
  cosK = constrain(cosK, -1.0, 1.0);
  float kneeRad = acos(cosK);

  float phi   = atan2(x, y);
  float delta = atan2(L_KNEE * sin(kneeRad), L_HIP + L_KNEE * cos(kneeRad));

  hipDeg  = degrees(phi - delta);
  kneeDeg = degrees(kneeRad);
  return true;
}

void setFoot(int id, float x, float y) {
  float hd, kd;
  if (!ikSolve(x, y, hd, kd)) return;

  LegDef &l = legs[id];
  float hipServo  = l.hip_trim  + l.hip_dir  * hd;
  float kneeServo = l.knee_trim - l.knee_dir * kd;

  writeServo(l.pca, l.hip_ch,  constrain(hipServo,  0.0, 180.0));
  writeServo(l.pca, l.knee_ch, constrain(kneeServo, 0.0, 180.0));
}

// ── Trot Gait "01 / 10" – 10-point semicircle ────────────────────────────────
void trotUpdate() {
  const int pairA[2] = { 0, 3 }; // FL, RR
  const int pairB[2] = { 1, 2 }; // FR, RL

  const int *swing  = (gPhase == 0) ? pairA : pairB;
  const int *stance = (gPhase == 0) ? pairB : pairA;

  for (int i = 0; i < 2; i++) {
    setFoot(swing[i],  swingX[gStep],  swingY[gStep]);
    setFoot(stance[i], stanceX[gStep], stanceY[gStep]);
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin((int8_t)PIN_SDA, (int8_t)PIN_SCL);

  Adafruit_PWMServoDriver *boards[2] = { &pcaA, &pcaB };
  for (int b = 0; b < 2; b++) {
    boards[b]->begin();
    boards[b]->setOscillatorFrequency(OSC_FREQ);
    boards[b]->setPWMFreq(PWM_FREQ);
  }
  delay(100);

  // ── Hip servos – fixed angle, not IK-driven ──────────────────────────────
  pcaA.setPWM(10, 0, deg2pwm(HIP_FL));
  pcaA.setPWM(11, 0, deg2pwm(HIP_FR));
  pcaB.setPWM(15, 0, deg2pwm(HIP_RL));
  pcaB.setPWM(14, 0, deg2pwm(HIP_RR));

  precomputeGait();

  Serial.println("Standing...");
  for (int i = 0; i < 4; i++) setFoot(i, 0.0, STANCE_H);
  delay(1500);

  Serial.println("Walking.");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  trotUpdate();

  gStep++;
  if (gStep >= GAIT_STEPS) {
    gStep = 0;
    gPhase ^= 1;   // toggle swing/stance pairs
  }

  // Step period controls speed: 10 steps * period = half-cycle duration
  // At SPEED_HZ=1.0 → 100ms per step → 1 full stride per 2 seconds
  delay((unsigned long)(1000.0 / (GAIT_STEPS * 2.0 * SPEED_HZ)));
}