using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

###############################################################################
# semidiscretization of the Maxwell equations: transmittance and reflectance of
# a dielectric slab

equations = MaxwellEquations3D(Heterogeneous())

slab_thickness = 0.5
slab_material = Material(epsilon = 2.25)
material_at(x) = 0 < x[3] < slab_thickness ? slab_material : Material()

# Gaussian plane-wave pulse along z, polarized in x, in front of the slab
incident_field = PlaneWave((0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                           GaussianPulse(0.1; delay = 0.8))
initial_condition = incident_field

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
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# the multiple reflections inside the slab have decayed by the end
tspan = (0.0, 6.0)
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
                                    filename = "slab")

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
