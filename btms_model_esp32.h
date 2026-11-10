/*
 * BTMS ESP32 Edge Decision Engine
 * Auto-generated C++ code for embedded inference
 */

#ifndef BTMS_ML_MODEL_H
#define BTMS_ML_MODEL_H

// Thresholds derived from trained XGBoost Classifier
const float TEMP_GRADIENT_THRESHOLD = 2.8819;
const float TEMP_MAX_THRESHOLD      = 62.7128;

struct BatteryTelemetry {
    float voltage;
    float current;
    float temperature;
    float temp_rolling_mean;
    float temp_gradient;
};

enum BTMS_State {
    NORMAL_OPERATION = 0,
    PREEMPTIVE_COOLING_PUMP_ON = 1,
    THERMAL_RUNAWAY_EMERGENCY = 2
};

inline BTMS_State evaluate_btms_state(const BatteryTelemetry& data) {
    if (data.temp_gradient > TEMP_GRADIENT_THRESHOLD || data.temperature > TEMP_MAX_THRESHOLD) {
        return THERMAL_RUNAWAY_EMERGENCY;
    } else if (data.temp_gradient > (TEMP_GRADIENT_THRESHOLD * 0.5f)) {
        return PREEMPTIVE_COOLING_PUMP_ON;
    }
    return NORMAL_OPERATION;
}

#endif // BTMS_ML_MODEL_H
