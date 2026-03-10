#include <Arduino.h>

const int potPins[3] = {A0, A1, A2};
const int buttonPin = A3;
const int sampleCount = 10;
const int potRawMin[3] = {0, 0, 0};
const int potRawMax[3] = {1023, 1023, 1023};
const int degMin = 0;
const int degMax = 300;
const unsigned long frameIntervalMs = 20;  // 50Hz
unsigned long lastFrameMs = 0;

int readPotAvg(int pin) {
  long sum = 0;
  for (int i = 0; i < sampleCount; i++) {
    sum += analogRead(pin);
    delayMicroseconds(200);
  }
  return (int)(sum / sampleCount);
}

void setup() {
  Serial.begin(115200);
  pinMode(buttonPin, INPUT_PULLUP);
}

void loop() {
  unsigned long now = millis();
  if (now - lastFrameMs < frameIntervalMs) {
    return;
  }
  lastFrameMs = now;

  int raw[3];
  int deg[3];
  for (int i = 0; i < 3; i++) {
    raw[i] = readPotAvg(potPins[i]);
    int clipped = constrain(raw[i], potRawMin[i], potRawMax[i]);
    deg[i] = map(clipped, potRawMin[i], potRawMax[i], degMin, degMax);
  }
  int btn = (digitalRead(buttonPin) == LOW) ? 1 : 0;

  // Control line:
  // CTRL,<pot0_deg>,<pot1_deg>,<pot2_deg>,<grip>
  Serial.print("CTRL,");
  Serial.print(deg[0]);
  Serial.print(",");
  Serial.print(deg[1]);
  Serial.print(",");
  Serial.print(deg[2]);
  Serial.print(",");
  Serial.println(btn);
}
