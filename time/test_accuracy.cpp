#include <iostream>
#include <iomanip>
#include <qd/dd_real.h>

int main() {
    // We use a string to ensure we get a full 31-digit Pi
    // This forces the value to overflow the High Part into the Low Part
    dd_real my_pi("3.1415926535897932384626433832795");

    std::cout << "--- PI Accuracy Test ---" << std::endl;
    // .x[0] is the High Part, .x[1] is the Low Part
    std::cout << "High Part (.x[0]): " << std::setprecision(18) << my_pi.x[0] << std::endl;
    std::cout << "Low Part  (.x[1]): " << std::setprecision(18) << my_pi.x[1] << std::endl;
    
    std::cout << "\n--- Full Reconstruction ---" << std::endl;
    std::cout << "Full DD:           " << std::setprecision(32) << my_pi << std::endl;

    // Standard double comparison
    double d_pi = 3.14159265358979323846;
    std::cout << "Standard Double:   " << std::setprecision(32) << d_pi << std::endl;

    return 0;
}
