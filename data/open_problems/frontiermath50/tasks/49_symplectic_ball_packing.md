# Symplectic Ball Packing

- **Problem ID**: `symplectic-ball-packing`
- **Field**: Topology/Geometry
- **Notability**: Major advance
- **Tags**: Construction - Family
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Fix $n=2$. Let $k \ge 10$ be an integer and take $\epsilon \in (0,1)$. For $i = 1,\dots,k$, let $B_i \subset \mathbb{R}^{2n}$ denote the closed unit ball centered at $(3i-3,0,\dots,0)$.

Your goal is to find an algorithm that takes $k$ and $\epsilon$ as inputs and produces as output an explicit smooth Hamiltonian function $H_\epsilon: \mathbb{R}^{2n} \times [0,1] \rightarrow \mathbb{R}$ whose time-$1$ Hamiltonian flow $\phi: \mathbb{R}^{2n} \rightarrow \mathbb{R}^{2n}$ satisfies $\phi(\sqcup_{i=1}^{k}B_i) \subset B^{2n}(R)$, where $B^{2n}(R) \subset \mathbb{R}^{2n}$ is the closed ball centered at the origin of radius $R$, and where we require the volume ratio of $\bigsqcup\limits_{i=1}^{k} B_i$ to $B^{2n}(R)$ to satisfy $\dfrac{k}{R^{2n}} > 1 - \epsilon$.

A solution to this problem is meant to rest on finding a family of embeddings that has a natural algorithmic dependence on $\epsilon$. As such, it should be a single algorithm that inputs $k$ and $\epsilon$ and outputs the Hamiltonian $H_\epsilon$, rather than a separate ad-hoc construction for each value.

Submission format:
Submit a single Python file that defines a function `solution(k, epsilon)`. Given an integer $k \ge 10$ and a float `epsilon` in $(0,1)$, `solution(k, epsilon)` must return a string containing the source code of a Python module that defines a function `Hamiltonian(Q, P, t)` implementing $H_\epsilon$ with the following properties.
    - Takes inputs Q (positions), P (momenta), and t (time).
    - Must be implemented using PyTorch operations to support automatic differentiation.
    - Inputs Q and P will be `torch.Tensor`s of dtype `torch.float64` and shape `(N, n)` (where N is the size of a batch of points sampled from the boundaries of the input balls). Any internal parameters must be float64 as well (or cast to `Q.dtype`), or PyTorch will raise a dtype-mismatch error.
    - Input t will be a scalar float.
    - Returns a `torch.Tensor` of shape `(N,)` representing the Hamiltonian value for each point in the batch.
    - Is smooth, to ensure existence and uniqueness of flow.

Verification: your `solution(k, epsilon)` will be called for several values of $k \ge 10$. For each, the returned Hamiltonian is verified by sampling points on the boundaries of the balls, numerically integrating their flow under $H_\epsilon$ (via RK4, using PyTorch autograd for the gradient of $H_\epsilon$), and checking $\frac{k}{R^{2n}} > 1 - \epsilon$ where $R$ is the radius of the smallest ball containing the flowed points. For each $k$, the call to `solution(k, epsilon)` must finish within 10 minutes, and the resulting `Hamiltonian` function must return within 10 seconds on each call. The verifier also confirms that the flow it computed is symplectic: it tracks the symplectic form on pairs of nearby sampled points and rejects a submission whose reported gradients do not preserve it. A genuine Hamiltonian satisfies this automatically.

Two example files are provided. `solution_example.py` shows the required `solution(k, epsilon)` format (it returns a trivial Hamiltonian that does not solve the problem). `hamiltonian_example.py` is an example of the kind of `Hamiltonian` module `solution` should return; it is a partially-trained neural network for $k=4$.
