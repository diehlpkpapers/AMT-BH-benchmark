#pragma once
#include <iostream>
#include <string>

// parses time strings like "1h", "2.5d", "1y" into days
double parseTime(const std::string& timeStr) {
    double timeValue = std::stod(timeStr.substr(0, timeStr.size() - 1)); 
    char unit = timeStr.back(); // get the unit character
    switch (unit) {
        case 'h': return timeValue / 24.0;         // convert hours to days
        case 'd': return timeValue;                // already in days
        case 'y': return timeValue * 365.25;       // convert years to days
        default:
            throw std::invalid_argument("unknown time unit: " + timeStr);
    }
}
