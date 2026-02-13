#pragma once

#include <iostream>
#include <limits>
#include <random>
#include <complex>
#include <cassert>

const int CacheSize = 16;
static const double factorial_cache[CacheSize] = {
    1, 1, 2, 6, 24, 120, 720, 5040,
    40320, 362880, 3628800, 39916800,
    479001600, 6227020800,
    87178291200, 1307674368000
};

static const double double_factorial_cache[CacheSize] = {
    1, 1, 2, 3, 8, 15, 48, 105,
    384, 945, 3840, 10395, 46080,
    135135, 645120, 2027025
};


inline double Factorial(int x) 
{
  if (x < CacheSize) {
    return factorial_cache[x];
  } else {
    double s = factorial_cache[CacheSize - 1];
    for (int n = CacheSize; n <= x; n++) {
      s *= n;
    }
    return s;
  }
}

inline double DoubleFactorial(int x) 
{
  if (x < CacheSize) {
    return double_factorial_cache[x];
  } else {
    double s = double_factorial_cache[CacheSize - (x % 2 == 0 ? 2 : 1)];
    double n = x;
    while (n >= CacheSize) {
      s *= n;
      n -= 2.0;
    }
    return s;
  }
}


inline double AssociatedLegendreP(int l, int m, double x) {
    assert(l >= 0);
    assert(m >= 0);
    assert(std::abs(x) <= 1.0);

    double pmm = 1.0;
    if (m > 0) {
        double sqrt1mx2 = std::sqrt(1.0 - x * x);
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * sqrt1mx2;
            fact += 2.0;
        }
    }

    if (l == m) return pmm;

    double pmmp1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) return pmmp1;

    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }

    return pll;
}


// scipy 1.10.x 
// define m     => Order of the harmonic (int); must have |m| <= n.
// define n     => Degree of the harmonic (int); must have n >= 0. This is often denoted by l (lower case L) in descriptions of spherical harmonics.
// define theta => Azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
// define phi   => Polar (colatitudinal) coordinate; must be in [0, pi].
// in our flash code, use theta => polar coord and phi => azimuth coord, use sph_harm(m, n, phi, theta)
inline std::complex<double> sph_harm(int m, int n, double theta, double phi)  
{
    assert(n >= 0);
    assert(std::abs(m) <= n);

    if (m < 0) {
        return std::pow(-1.0, m) * std::conj(sph_harm(-m, n, theta, phi));
    }


    // int mp;
    // double prefactor = 1.0f;

    double x = std::cos(phi);  // for AssociatedLegendrePf
    double Pnm = AssociatedLegendreP(n, m, x);

    double norm = std::sqrt(
        ((2.0 * n + 1.0) / (4.0 * M_PI)) *
        (Factorial(n - m) / Factorial(n + m))
    );

    std::complex<double> e_imtheta = std::polar(1.0, m * theta);
    return norm * Pnm * e_imtheta;
}
