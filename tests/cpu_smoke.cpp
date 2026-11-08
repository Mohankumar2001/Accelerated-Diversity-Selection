#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    constexpr std::int64_t n = 100000;
    std::vector<std::int64_t> values(n);
    std::iota(values.begin(), values.end(), std::int64_t{1});
    const auto sum = std::accumulate(values.begin(), values.end(), std::int64_t{0});
    const auto expected = n * (n + 1) / 2;
    if (sum != expected) {
        std::cerr << "C++ check: FAIL (sum=" << sum << ")\n";
        return 1;
    }
    std::cout << "C++ compile and execution: PASS (sum=" << sum << ")\n";
}
