#include <iostream>
#include <cmath>
#include <string>

void check(float g_logit, float dot, int t, int v) {
    float ldiff = std::abs(g_logit - dot);
    float rel_diff = ldiff / std::max(1.0f, std::abs(dot));
    std::cout << "t=" << t << " v=" << v << " GPU=" << g_logit << " CPU=" << dot
              << " diff=" << ldiff << " rel=" << rel_diff << std::endl;
}
