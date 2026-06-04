#include <sys/utsname.h>

#include <iostream>

// a small helper to display the content of an utsname struct:
std::ostream& operator<<(std::ostream& os, const utsname& u) {
    return os << "sysname : " << u.sysname << '\n'
              << "nodename: " << u.nodename << '\n'
              << "release : " << u.release << '\n'
              << "version : " << u.version << '\n'
              << "machine : " << u.machine << '\n';
}

int main() {
    utsname result;      // declare the variable to hold the result

    uname(&result);      // call the uname() function to fill the struct

    std::cout << result; // show the result using the helper function
}
