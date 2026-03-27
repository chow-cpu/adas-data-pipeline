#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>

struct VehicleState {
    double timestamp;
    double speed_mps;
    double accel_x;
    double accel_y;
    double accel_z;
    double steering_angle;
    double radar_distance_m;
    double latitude;
    double longitude;
};

double addNoise(double value, double noiseLevel) {
    double noise = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    return value + noise * noiseLevel;
}

void simulateVehicle(std::string filename, std::string profile,
                     double startLat, double startLon) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        return;
    }

    file << "timestamp,speed_mps,accel_x,accel_y,accel_z,"
         << "steering_angle,radar_distance_m,latitude,longitude\n";

    double lat = startLat;
    double lon = startLon;
    double baseSpeed, radarBase;

    if (profile == "highway") {
        baseSpeed = 30.0;
        radarBase = 80.0;
    } else if (profile == "city") {
        baseSpeed = 8.0;
        radarBase = 15.0;
    } else {
        baseSpeed = 20.0;
        radarBase = 40.0;
    }

    for (int i = 0; i < 100; i++) {
        double timestamp = i * 0.1;

        // Aggressive profile anomalies
        if (profile == "aggressive") {
            if (i == 30) baseSpeed = 45.0;
            else if (i == 31) baseSpeed = 20.0;
            else if (i == 70) baseSpeed = 0.5;
            else if (i == 71) baseSpeed = 20.0;
        }

        // City profile stop and go
        if (profile == "city") {
            if (i % 20 == 0) baseSpeed = 2.0;
            else baseSpeed = 8.0;
        }

        lat += 0.0001 + addNoise(0.0, 0.00002);
        lon += 0.0001 + addNoise(0.0, 0.00002);

        VehicleState state;
        state.timestamp        = timestamp;
        state.speed_mps        = addNoise(baseSpeed, 0.3);
        state.accel_x          = addNoise(0.2, 0.1);
        state.accel_y          = addNoise(-0.1, 0.05);
        state.accel_z          = addNoise(9.8, 0.1);
        state.steering_angle   = addNoise(2.0, 0.5);
        state.radar_distance_m = addNoise(radarBase, 1.0);
        state.latitude         = lat;
        state.longitude        = lon;

        file << state.timestamp        << ","
             << state.speed_mps        << ","
             << state.accel_x          << ","
             << state.accel_y          << ","
             << state.accel_z          << ","
             << state.steering_angle   << ","
             << state.radar_distance_m << ","
             << state.latitude         << ","
             << state.longitude        << "\n";
    }

    file.close();
    std::cout << "Generated: " << filename << std::endl;
}

int main() {
    srand(time(0));

    std::cout << "=== ADAS Multi-Vehicle Simulator ===" << std::endl;

    // Three vehicles starting at slightly different positions in Detroit
    simulateVehicle("data/vehicle_a.csv", "highway", 42.3314, -83.0458);
    simulateVehicle("data/vehicle_b.csv", "city",    42.3300, -83.0470);
    simulateVehicle("data/vehicle_c.csv", "aggressive", 42.3330, -83.0445);

    std::cout << "All vehicles simulated successfully!" << std::endl;
    return 0;
}