@doc raw"""
    mie_efficiencies(m, x)

Scattering, extinction and absorption efficiencies ``Q = \sigma / (\pi a^2)`` of a
homogeneous sphere of radius ``a`` with relative refractive index `m` and size
parameter `x` ``= k a`` in the background medium, from the Mie series in the form
of Bohren and Huffman (`BHMIE`). An absorbing sphere has ``\operatorname{Im} m > 0``.
Returns a named tuple `(; scattering, extinction, absorption)`.
"""
function mie_efficiencies(m::Number, x::Real)
    RealT = float(typeof(x))
    num_terms = round(Int, x + 4 * cbrt(x) + 2)
    y = m * x
    num_logderivative = max(num_terms, round(Int, abs(y))) + 15

    # logarithmic derivative D_n(y) by downward recurrence, D[n + 1] = D_n
    logderivative = zeros(complex(typeof(y)), num_logderivative + 1)
    for n in num_logderivative:-1:1
        logderivative[n] = n / y - 1 / (logderivative[n + 1] + n / y)
    end

    psi_previous, psi_current = cos(x), sin(x)
    chi_previous, chi_current = -sin(x), cos(x)
    xi_current = complex(psi_current, -chi_current)
    scattering = zero(RealT)
    extinction = zero(RealT)
    for n in 1:num_terms
        psi = (2 * n - 1) * psi_current / x - psi_previous
        chi = (2 * n - 1) * chi_current / x - chi_previous
        xi = complex(psi, -chi)
        D = logderivative[n + 1]
        a = ((D / m + n / x) * psi - psi_current) / ((D / m + n / x) * xi - xi_current)
        b = ((m * D + n / x) * psi - psi_current) / ((m * D + n / x) * xi - xi_current)
        scattering += (2 * n + 1) * (abs2(a) + abs2(b))
        extinction += (2 * n + 1) * real(a + b)
        psi_previous, psi_current = psi_current, psi
        chi_previous, chi_current = chi_current, chi
        xi_current = xi
    end
    scattering *= 2 / x^2
    extinction *= 2 / x^2
    return (; scattering, extinction, absorption = extinction - scattering)
end
