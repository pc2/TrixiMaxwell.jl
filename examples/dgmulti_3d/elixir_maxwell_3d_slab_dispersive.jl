using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

###############################################################################
# semidiscretization of the Maxwell equations: transmittance and reflectance of
# a slab with one Drude and one Lorentz pole

# one inactive pole of each kind as the default, the slab activates them
equations = MaxwellEquations3D(Heterogeneous(); drude = (DrudePole(0.0, 0.0),),
                               lorentz = (LorentzPole(0.0, 0.0, 0.0),))

slab_thickness = 0.5
# metallic below the plasma frequency f = 1, a damped resonance at f = 1.5
slab_material = Material(epsilon = 1.5, drude = (DrudePole(2 * pi, 1.0),),
                         lorentz = (LorentzPole(1.0, 3 * pi, 2.0),))
material_at(x) = 0 < x[3] < slab_thickness ? slab_material : Material()

# Gaussian plane-wave pulse along z, polarized in x, in front of the slab
incident_field = PlaneWave((0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                           GaussianPulse(0.1; delay = 0.8))
initial_condition = incident_field

source_terms = source_terms_dispersive

polydeg = 3
surface_flux = flux_upwind
solver = DGMulti(polydeg = polydeg,
                 element_type = Tet(),
                 approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(surface_flux),
                 volume_integral = VolumeIntegralWeakForm())

# periodic in x and y; the slab faces and the detector planes are element faces
cells_per_dimension = (2, 2, 28)
mesh = DGMultiMesh(solver, cells_per_dimension;
                   coordinates_min = (0.0, 0.0, -1.5),
                   coordinates_max = (0.25, 0.25, 2.0),
                   periodicity = (true, true, false))

# Silver-Mueller absorbs plane waves at normal incidence
boundary_conditions = (; entire_boundary = boundary_condition_silver_mueller)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions, source_terms)

###############################################################################
# ODE solvers, callbacks etc.

# the polarization currents have decayed by the end
tspan = (0.0, 8.0)
ode = semidiscretize(semi, tspan)
set_materials!(ode.u0, semi, material_at)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[])
alive_callback = AliveCallback(analysis_interval = analysis_interval)

frequencies = range(0.25, 2.0, length = 15)
detector_front = DetectorPlaneCallback(semi, incident_field, frequencies;
                                       point = (0.0, 0.0, -0.5), normal = (0.0, 0.0, 1.0))
detector_back = DetectorPlaneCallback(semi, incident_field, frequencies;
                                      point = (0.0, 0.0, 1.0), normal = (0.0, 0.0, 1.0))

save_vtk_callback = SaveVtkCallback(dt = 0.1, output_directory = "out",
                                    filename = "slab_dispersive")

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        detector_front, detector_back, save_vtk_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # overwritten by the stepsize callback
            ode_default_options()..., callback = callbacks)

reflectance = transmittance_reflectance(detector_front).reflectance
transmittance = transmittance_reflectance(detector_back).transmittance
