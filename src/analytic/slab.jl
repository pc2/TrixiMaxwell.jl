@doc raw"""
    slab_transmittance_reflectance(n, thickness, frequency)

Power transmittance and reflectance of a slab with relative refractive index `n`
and the given `thickness` in vacuum at normal incidence (Airy formula),
```math
t = \frac{t_{12} t_{21} e^{i \delta}}{1 - r_{21}^2 e^{2 i \delta}},
\quad
r = r_{12} + \frac{t_{12} t_{21} r_{21} e^{2 i \delta}}{1 - r_{21}^2 e^{2 i \delta}},
\quad
\delta = 2 \pi f n d,
```
with the interface coefficients ``r_{12} = (1 - n) / (1 + n) = -r_{21}``,
``t_{12} = 2 / (1 + n)`` and ``t_{21} = 2 n / (1 + n)``. An absorbing slab has
``\operatorname{Im} n > 0``. Returns `(; transmittance, reflectance)`.
"""
function slab_transmittance_reflectance(n::Number, thickness::Real, frequency::Real)
    r12 = (1 - n) / (1 + n)
    r21 = -r12
    t12 = 2 / (1 + n)
    t21 = 2 * n / (1 + n)
    phase = cis(2 * oftype(float(frequency), pi) * frequency * n * thickness)
    denominator = 1 - r21^2 * phase^2
    t = t12 * t21 * phase / denominator
    r = r12 + t12 * t21 * r21 * phase^2 / denominator
    return (; transmittance = abs2(t), reflectance = abs2(r))
end
