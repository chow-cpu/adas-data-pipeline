#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>

// Represents the current state of a simulated vehicle
struct VehicleState {
    double timestamp;
    double speed_mps;
    double accel_x;
    double accel_y;
    double accel_z;
    double steering_angle;
    double radar_distance_m;
};

// Add small random noise to simulate real sensor imperfection
double addNoise(double value, double noiseLevel) {
    double noise = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    return value + noise * noiseLevel;
}

// Simulate a vehicle driving for a given number of steps
VehicleState simulateStep(double timestamp, double baseSpeed) {
    VehicleState state;
    state.timestamp        = timestamp;
    state.speed_mps        = addNoise(baseSpeed, 0.3);
    state.accel_x          = addNoise(0.2, 0.1);
    state.accel_y          = addNoise(-0.1, 0.05);
    state.accel_z          = addNoise(9.8, 0.1);
    state.steering_angle   = addNoise(2.0, 0.5);
    state.radar_distance_m = addNoise(45.0 - baseSpeed, 1.0);
    return state;
}

int main() {
    srand(time(0));

    std::string outputPath = "data/simulated_sensor_log.csv";
    std::ofstream file(outputPath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file." << std::endl;
        return 1;
    }

    // Write CSV header
    file << "timestamp,speed_mps,accel_x,accel_y,accel_z,"
         << "steering_angle,radar_distance_m\n";

    std::cout << "=== ADAS Sensor Simulator ===" << std::endl;
    std::cout << "Generating 100 sensor readings..." << std::endl;

    double baseSpeed = 10.0;

    for (int i = 0; i < 100; i++) {
        double timestamp = i * 0.1;

        // Simulate a sudden speed spike at step 50 (anomaly)
        if (i == 50) baseSpeed = 25.0;
        else if (i == 51) baseSpeed = 10.0;

        VehicleState state = simulateStep(timestamp, baseSpeed);

        file << state.timestamp        << ","
             << state.speed_mps        << ","
             << state.accel_x          << ","
             << state.accel_y          << ","
             << state.accel_z          << ","
             << state.steering_angle   << ","
             << state.radar_distance_m << "\n";
    }

    file.close();
    std::cout << "Done! Data saved to " << outputPath << std::endl;
    return 0;
}