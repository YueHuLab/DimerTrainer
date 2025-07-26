Dimer-Enhanced Optimization: A First-Order Approach to Escaping Saddle Points in Neural Network Training



First-order optimization methods, such as SGD and Adam, are widely used for
training large-scale deep neural networks due to their computational efficiency and
robust performance. However, relying solely on gradient information, these methods
often struggle to navigate complex loss landscapes featuring flat regions, plateaus,
and saddle points. Second-order methods, which utilize curvature information from
the Hessian matrix, can address these challenges but are computationally infeasible
for large models. The Dimer method, a first-order technique that constructs two
closely spaced points to probe the local geometry of a potential energy surface,
offers an efficient approach to estimate curvature using only gradient information.
Drawing inspiration from its application in molecular dynamics simulations for locating
saddle points, we propose Dimer-Enhanced Optimization (DEO), a novel
framework to facilitate escaping saddle points in neural network training. Unlike
its original use, DEO adapts the Dimer method to explore a broader region of the
loss landscape, efficiently approximating the Hessian’s smallest eigenvector without
computing the full Hessian matrix. By periodically projecting the gradient
onto the subspace orthogonal to the minimum curvature direction, DEO guides the
optimizer away from saddle points and flat regions, promoting more effective training
while significantly reducing sampling time costs through non-stepwise updates.
Preliminary experiments on Transformer-based toy models show that DEO achieves
competitive performance compared to standard first-order methods, enabling improved
navigation of complex loss landscapes. Our work offers a practical approach
to repurpose physics-inspired, first-order curvature estimation for enhancing neural
network training in high-dimensional spaces.
