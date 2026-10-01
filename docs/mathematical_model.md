# Mathematical model

For an active stack of layers, the semi-discrete thermal model is

\[
\dot{x}(t) = A x(t) + B(t)u(t) + d,
\qquad y(t) = C(t)x(t).
\]

Each layer is a rectangular four-neighbour grid. The graph Laplacian describes
in-plane conduction. Off-diagonal identity blocks describe conduction between
matching cells in adjacent layers. The newest layer is powder; older layers
are treated as dense material. Ambient convection acts on the top layer and
the substrate boundary condition acts on the bottom layer.

The moving laser profile is a spatial Gaussian with variance
\(R^2/9\) and absorptivity \(\alpha\). Direct simulation and evaluation
constrain commanded power to \([0,p_{\max}]\). Specialized multilayer training
uses an unconstrained candidate power with a soft penalty outside that interval
before evaluating the trained controller with physical clipping. The
observation vector retains the spatial weighting used in the original
implementation so that old and new experiments can be compared.

The continuous matrices are assembled as

\[
A=C_h^{-1}K,\qquad B(t)=C_h^{-1}b(t),\qquad d=C_h^{-1}d_{\mathrm{physical}},
\]

where \(C_h\) is the diagonal cell heat-capacity matrix and \(K\) is the
coupled conductance matrix.

For long builds, the simulator keeps a fixed number of active ROI layers. When
a new layer would exceed this count, the two oldest temperature fields are
averaged into one merged bottom field. This approximation is explicit in the
simulation code and should be documented in any reported experiment.
