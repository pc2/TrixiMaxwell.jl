module TrixiMaxwell

using StaticArrays: SVector, SMatrix
using LinearAlgebra: norm, dot, cross

import Trixi

using Trixi: DGMulti, DGMultiMesh, DGSEM, eachnode, entropy_timederivative, energy_total
using Trixi: StartUpDG
using WriteVTK: vtk_grid, vtk_save, MeshCell, VTKCellTypes, VTKCellData

include("equations/maxwell_3d.jl")
include("equations/materials.jl")
include("callbacks_step/analysis_dgmulti.jl")
include("callbacks_step/analysis_dgsem.jl")
include("solvers/dgsem_heterogeneous.jl")
include("meshes/imported_mesh.jl")
include("meshes/gambit.jl")
include("meshes/gmsh.jl")
include("meshes/download.jl")
include("visualization/vtk.jl")
include("visualization/point_evaluation.jl")
include("sources/signals.jl")
include("sources/incident_fields.jl")
include("sources/dipole.jl")
include("sources/tfsf.jl")
include("sources/tfsf_dgsem.jl")
include("sources/pml.jl")
include("sources/projected.jl")
include("callbacks_step/save_vtk.jl")
include("callbacks_step/surface_transforms.jl")
include("analytic/mie.jl")
include("analytic/slab.jl")

export MaxwellEquations3D, Homogeneous, Heterogeneous, NoPML, UPML,
       NonDispersive, Dispersive, DrudePole, LorentzPole, source_terms_dispersive,
       relative_permittivity,
       FluxUpwindPenalty, flux_upwind,
       permittivity, permeability, conductivity, impedance, admittance, speed_of_light,
       source_terms_conductivity, Material, set_materials!,
       boundary_condition_perfect_electric_conductor,
       boundary_condition_perfect_magnetic_conductor,
       boundary_condition_silver_mueller, BoundaryConditionIncidentField,
       initial_condition_cavity,
       ImportedMesh, read_gambit, read_gmsh, download_mesh,
       write_mesh_vtk, write_solution_vtk, SaveVtkCallback, PointEvaluator,
       GaussianPulse, ModulatedGaussianPulse, signal_derivative,
       signal_second_derivative, PlaneWave, initial_condition_zero,
       HertzianDipole, HertzianDipoleField, TotalFieldScatteredField,
       FluxTotalFieldScatteredField,
       PMLProfile, SourceTermsPML, CombinedSourceTerms, ProjectedSourceTerms,
       CrossSectionCallback, DetectorPlaneCallback, cross_sections,
       transmittance_reflectance, mie_efficiencies, slab_transmittance_reflectance

end
