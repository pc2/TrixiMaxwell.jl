# ParaView setup for the output of examples/dgmulti_3d/elixir_maxwell_3d_slab.jl
# and elixir_maxwell_3d_slab_dispersive.jl: 3D view colored by Ex with the slab
# outlined, and Ex, Hy and epsilon (or the Lorentz polarization) along z.
# Run from the package root with
#   paraview --script=utils/paraview/slab.py
# or set SLAB_PVD to the path of slab.pvd or slab_dispersive.pvd.
import os

from paraview.simple import *

path = os.path.abspath(os.environ.get("SLAB_PVD", os.path.join("out", "slab.pvd")))
reader = PVDReader(registrationName="slab.pvd", FileName=path)
reader.UpdatePipelineInformation()

render_view = GetActiveViewOrCreate("RenderView")
layout = GetLayout(render_view)
if layout is None:
    layout = CreateLayout("slab")
    AssignViewToLayout(view=render_view, layout=layout, hint=0)

display = Show(reader, render_view)
display.NonlinearSubdivisionLevel = 2
ColorBy(display, ("POINTS", "Ex"))
lut = GetColorTransferFunction("Ex")
lut.ApplyPreset("Cool to Warm", True)
lut.AutomaticRescaleRangeMode = "Never"
lut.RescaleTransferFunction(-1.0, 1.0)
display.SetScalarBarVisibility(render_view, True)
scalar_bar = GetScalarBar(lut, render_view)
scalar_bar.WindowLocation = "Any Location"
scalar_bar.Position = [0.05, 0.3]
scalar_bar.HorizontalTitle = 1

# dispersive media are marked by their poles, the others by epsilon
point_arrays = reader.PointData.keys()
marker = next((name for name in ("omega_p_d1", "delta_epsilon_l1") if name in point_arrays),
              "epsilon")
slab = Threshold(registrationName="slab", Input=reader)
slab.Scalars = ["POINTS", marker]
slab.LowerThreshold = 1.01 if marker == "epsilon" else 1e-6
slab.UpperThreshold = 1e30
slab_display = Show(slab, render_view)
slab_display.SetRepresentationType("Outline")
ColorBy(slab_display, None)
slab_display.AmbientColor = [0.0, 0.0, 0.0]
slab_display.DiffuseColor = [0.0, 0.0, 0.0]

render_view.OrientationAxesVisibility = 1
render_view.ResetActiveCameraToPositiveX()
render_view.ResetCamera(False)

line = PlotOverLine(registrationName="line along z", Input=reader)
line.Point1 = [0.125, 0.125, -1.5]
line.Point2 = [0.125, 0.125, 2.0]
line.Resolution = 2000

layout.SplitHorizontal(0, 0.35)
chart = CreateView("XYChartView")
AssignViewToLayout(view=chart, layout=layout, hint=2)
line_display = Show(line, chart)
line_display.UseIndexForXAxis = 0
line_display.XArrayName = "Points_Z"
line_display.SeriesVisibility = ["Ex", "Hy",
                                 "epsilon" if marker == "epsilon" else "Px_l1"]
line_display.SeriesColor = ["Ex", "0.8", "0.1", "0.1", "Hy", "0.1", "0.3", "0.8",
                            "epsilon", "0.5", "0.5", "0.5", "Px_l1", "0.2", "0.6", "0.2"]
chart.BottomAxisTitle = "z"
chart.LeftAxisTitle = ""
chart.LeftAxisUseCustomRange = 1
chart.LeftAxisRangeMinimum = -1.0
chart.LeftAxisRangeMaximum = 2.5

GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
SetActiveView(chart)
