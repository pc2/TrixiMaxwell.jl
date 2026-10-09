const SPEED_OF_LIGHT_VACUUM = 299_792_458.0

"""
    normalized_angular_frequency(frequency, length_unit)

Angular frequency `2π frequency` in hertz converted to the normalized units of
[`MaxwellEquations3D`](@ref) for lengths measured in `length_unit` meters.
"""
function normalized_angular_frequency(frequency, length_unit)
    return 2 * pi * frequency * length_unit / SPEED_OF_LIGHT_VACUUM
end

@doc raw"""
    material_gold(length_unit)

Gold as one [`DrudePole`](@ref) and one [`LorentzPole`](@ref), fitted to the
Johnson and Christy data between 500 and 1000 nm (1.24 to 2.48 eV), for lengths
measured in `length_unit` meters, for example `1e-9` for nanometers.
- Alexandre Vial, Anne-Sophie Grimault, Demetrio Macías, Dominique Barchiesi,
  Marc Lamy de la Chapelle (2005)
  Improved analytical fit of gold dispersion: Application to the modeling of
  extinction spectra with a finite-difference time-domain method
  [DOI: 10.1103/PhysRevB.71.085416](https://doi.org/10.1103/PhysRevB.71.085416)
"""
function material_gold(length_unit)
    omega(frequency_thz) = normalized_angular_frequency(frequency_thz * 1e12, length_unit)
    return Material(epsilon = 5.9673,
                    drude = (DrudePole(omega(2113.6), omega(15.92)),),
                    lorentz = (LorentzPole(1.09, omega(650.07), omega(104.86)),))
end
