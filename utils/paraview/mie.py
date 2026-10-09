# ParaView setup for the output of examples/dgmulti_3d/elixir_maxwell_3d_mie.jl
# and elixir_maxwell_3d_mie_drude.jl: Ex on the plane y = 0 with the sphere and
# the TF/SF box outlined, and Ex, Hy and, for a dielectric sphere, epsilon along
# the z axis.
# Run from the package root with
#   paraview --script=utils/paraview/mie.py
# or set MIE_PVD to the path of mie.pvd or mie_drude.pvd.
import os

from paraview.simple import *

path = os.path.abspath(os.environ.get("MIE_PVD", os.path.join("out", "mie.pvd")))
reader = PVDReader(registrationName="mie.pvd", FileName=path)
reader.UpdatePipelineInformation()

render_view = GetActiveViewOrCreate("RenderView")
layout = GetLayout(render_view)
if layout is None:
    layout = CreateLayout("mie")
    AssignViewToLayout(view=render_view, layout=layout, hint=0)

# the scattered field outside the box is weaker than the incident pulse inside
field_range = 0.3
plane = Slice(registrationName="plane y = 0", Input=reader)
plane.SliceType = "Plane"
plane.SliceType.Origin = [0.0, 0.0, 0.0]
plane.SliceType.Normal = [0.0, 1.0, 0.0]
plane_display = Show(plane, render_view)
ColorBy(plane_display, ("POINTS", "Ex"))
lut = GetColorTransferFunction("Ex")
lut.ApplyPreset("Cool to Warm", True)
lut.AutomaticRescaleRangeMode = "Never"
lut.RescaleTransferFunction(-field_range, field_range)
plane_display.SetScalarBarVisibility(render_view, True)
scalar_bar = GetScalarBar(lut, render_view)
scalar_bar.WindowLocation = "Any Location"
scalar_bar.Position = [0.03, 0.3]
scalar_bar.HorizontalTitle = 1

sphere = Sphere(registrationName="sphere", Radius=0.5, ThetaResolution=128,
                PhiResolution=64)
circle = Slice(registrationName="sphere outline", Input=sphere)
circle.SliceType = "Plane"
circle.SliceType.Normal = [0.0, 1.0, 0.0]
box = Box(registrationName="TF/SF box", XLength=1.8, YLength=1.8, ZLength=1.8)
for source in (circle, box):
    outline_display = Show(source, render_view)
    outline_display.SetRepresentationType("Wireframe" if source is circle else "Outline")
    outline_display.AmbientColor = [0.0, 0.0, 0.0]
    outline_display.DiffuseColor = [0.0, 0.0, 0.0]
    outline_display.LineWidth = 2.0

render_view.OrientationAxesVisibility = 1
render_view.ResetActiveCameraToNegativeY()
render_view.ResetCamera(False)

# the line probe misses points in curved cells; sample linear subcells instead
linear_cells = Tessellate(registrationName="linear subcells", Input=reader)
linear_cells.MaximumNumberofSubdivisions = 4
linear_cells.ChordError = 1e-4
linear_cells.MergePoints = 0
line = PlotOverLine(registrationName="line along z", Input=linear_cells)
line.Point1 = [0.0, 0.0, -2.5]
line.Point2 = [0.0, 0.0, 2.5]
line.Resolution = 2000

layout.SplitHorizontal(0, 0.45)
chart = CreateView("XYChartView")
AssignViewToLayout(view=chart, layout=layout, hint=2)
line_display = Show(line, chart)
line_display.UseIndexForXAxis = 0
line_display.XArrayName = "Points_Z"
# epsilon marks a dielectric sphere; a Drude sphere keeps epsilon = 1
dispersive = "omega_p_d1" in reader.PointData.keys()
line_display.SeriesVisibility = ["Ex", "Hy"] if dispersive else ["Ex", "Hy", "epsilon"]
line_display.SeriesColor = ["Ex", "0.8", "0.1", "0.1", "Hy", "0.1", "0.3", "0.8",
                            "epsilon", "0.5", "0.5", "0.5"]
chart.BottomAxisTitle = "z"
chart.LeftAxisTitle = ""
chart.LeftAxisUseCustomRange = 1
chart.LeftAxisRangeMinimum = -1.0
chart.LeftAxisRangeMaximum = 2.5

GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
SetActiveView(chart)
