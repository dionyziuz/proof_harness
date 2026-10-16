"""Curated Example Open JSON Math Problems for the Prime Math Portal."""

from __future__ import annotations

from typing import Any, Dict, List

EXAMPLE_PROBLEMS: List[Dict[str, Any]] = [
    {
        "problem_id": "tcs-partition-character-nonvanishing",
        "title": "Non-Vanishing of Murnaghan-Nakayama Characters on Partition Products",
        "field": "Theoretical Computer Science & Algebraic Combinatorics",
        "tags": ["Murnaghan-Nakayama", "Symmetric Group", "Domino Tilings", "Staircases"],
        "notability": "TCS-Bench Benchmark Core",
        "literature_search": False,
        "max_iterations": 8,
        "context": (
            "Let \\lambda \\vdash n' be an integer partition. By the Murnaghan--Nakayama rule, the irreducible character "
            "\\chi_\\lambda of the symmetric group evaluated on cycle type \\rho equals the signed sum of border-strip tableaux:\n"
            "\\[ \\chi_\\lambda(\\rho) = \\sum_{T \\in \\mathcal{B}(\\lambda, \\kappa)} (-1)^{\\mathrm{ht}(T)} \\]\n"
            "for any ordering \\kappa of \\rho. For a finite set of partitions F, let \\Gamma_F denote the collection of skew shapes "
            "that admit a border strip tableau of format \\rho \\in F, and define\n"
            "\\[ \\alpha_F(\\gamma) = \\sum_{\\rho \\in F} \\sum_{T \\in \\mathcal{B}(\\gamma, \\rho)} (-1)^{\\mathrm{ht}(T)}. \\]\n"
            "Let F = \\{(2^2), (4)\\}. A size-4 skew shape \\gamma has \\alpha_F(\\gamma) \\neq 0 if and only if \\gamma is a non-vanishing tetromino, "
            "in which case \\alpha_F(\\gamma) = \\pm 2 according to the parity of its domino tilings.\n"
            "Let d(\\lambda) denote the maximum number of dominos peelable successively from \\lambda; the residual partition is a unique "
            "staircase \\mu = (k, k-1, \\dots, 1) of size z(\\lambda). Let s(\\lambda) be the maximum number of non-vanishing tetrominos "
            "peelable successively from \\lambda.\n"
            "For s \\le s(\\lambda), define the partition product multiset:\n"
            "\\[ \\theta_s := F^{\\times s} \\times \\{(2^{d(\\lambda)-2s}, 1^{z(\\lambda)})\\}. \\]"
        ),
        "hypotheses": [
            "lambda is a partition of n'",
            "s is a non-negative integer satisfying s <= s(lambda)",
            "d(lambda) >= 2s by domino refinability of non-vanishing tetrominos",
            "Terminal staircase mu = (k, k-1, ..., 1) is invariant under domino-peeling order",
        ],
        "statement": (
            "For any partition \\lambda \\vdash n' and integer s \\le s(\\lambda), the linearly extended symmetric group character satisfies:\n"
            "\\[ \\chi_\\lambda(\\theta_s) \\neq 0. \\]"
        ),
    },
    {
        "problem_id": "erdos-distinct-distances-planar",
        "title": "Extremal Bounds on Distinct Distances in Planar Point Sets",
        "field": "Discrete Geometry & Additive Combinatorics",
        "tags": ["Erdos Conjecture", "Incidence Geometry", "Crossing Lemma", "Polynomial Method"],
        "notability": "Fields-Level Breakthrough (Guth-Katz)",
        "literature_search": True,
        "max_iterations": 8,
        "context": (
            "Let P be a finite set of n points in the Euclidean plane \\mathbb{R}^2. The set of distinct pairwise distances is defined as:\n"
            "\\[ \\Delta(P) := \\{ \\|p - q\\|_2 : p, q \\in P, p \\neq q \\}. \\]\n"
            "In 1946, Paul Erd\\H{o}s conjectured that every n-point set in \\mathbb{R}^2 determines at least \\Omega(n / \\sqrt{\\log n}) distinct distances, "
            "matching the upper bound provided by the \\sqrt{n} \\times \\sqrt{n} integer lattice square grid.\n"
            "Using the Elekes--Sharir framework, the problem reduces to bounding the number of pairs of points with identical distances, "
            "which translates into line-line intersections in 3-dimensional space \\mathbb{R}^3."
        ),
        "hypotheses": [
            "P subset of R^2 is a finite set of cardinality |P| = n >= 3",
            "Euclidean L2 norm ||p - q||_2 is strictly metric and translation-invariant",
            "No more than O(n) points lie on any single line",
        ],
        "statement": (
            "Prove that there exists an absolute universal constant c > 0 such that for every set P of n points in \\mathbb{R}^2:\n"
            "\\[ |\\Delta(P)| \\ge c \\frac{n}{\\log n}. \\]"
        ),
    },
    {
        "problem_id": "baillie-psw-pseudoprime-bound",
        "title": "Non-Existence of Strong Lucas Pseudoprimes with Small Prime Factors",
        "field": "Computational Number Theory",
        "tags": ["Number Theory", "Primality Testing", "Lucas Sequences", "Recurrences"],
        "notability": "Open Conjectural Certificate (Epoch AI)",
        "literature_search": True,
        "max_iterations": 8,
        "context": (
            "The Baillie--PSW primality test combines a Miller--Rabin base-2 strong pseudoprime test with a strong Lucas pseudoprime test "
            "using the Selfridge parameters (D, P, Q) where D is the first element in 5, -7, 9, -11, ... with Jacobi symbol (D/n) = -1.\n"
            "Let U_k(P, Q) and V_k(P, Q) denote the Lucas sequence terms defined by:\n"
            "\\[ U_0 = 0, \\quad U_1 = 1, \\quad U_k = P U_{k-1} - Q U_{k-2}, \\]\n"
            "\\[ V_0 = 2, \\quad V_1 = P, \\quad V_k = P V_{k-1} - Q V_{k-2}. \\]\n"
            "An odd composite integer n is a Baillie--PSW pseudoprime if n is both a strong base-2 Fermat pseudoprime and a strong Lucas pseudoprime. "
            "No composite integer has ever been found that passes both tests."
        ),
        "hypotheses": [
            "n is an odd composite integer with n >= 3",
            "n satisfies 2^(d * 2^r) == 1 (mod n) or 2^(d * 2^s) == -1 (mod n) for n-1 = d * 2^s",
            "Jacobi symbol (D/n) = -1 with Selfridge discriminant choice D",
        ],
        "statement": (
            "Prove that if an odd composite integer n is a Baillie--PSW pseudoprime, then n must have at least 3 distinct prime factors, "
            "and all prime factors p | n must satisfy p > 2^{20}."
        ),
    },
    {
        "problem_id": "spectral-cheeger-mixing-gap",
        "title": "Cheeger Inequality Bounds for High-Girth Regular Expanders",
        "field": "Spectral Graph Theory & Theoretical Computer Science",
        "tags": ["Expander Graphs", "Cheeger Inequality", "Ramanujan Graphs", "Spectral Gap"],
        "notability": "Classical TCS & Extremal Graph Theory",
        "literature_search": False,
        "max_iterations": 8,
        "context": (
            "Let G = (V, E) be a connected d-regular undirected graph on n vertices. Let A denote the adjacency matrix of G, "
            "with eigenvalues d = \\lambda_1 > \\lambda_2 \\ge \\dots \\ge \\lambda_n \\ge -d.\n"
            "The edge expansion (Cheeger constant) h(G) is defined as:\n"
            "\\[ h(G) := \\min_{S \\subset V, 0 < |S| \\le n/2} \\frac{|e(S, V \\setminus S)|}{|S|}. \\]\n"
            "The discrete Cheeger inequality states that:\n"
            "\\[ \\frac{d - \\lambda_2}{2} \\le h(G) \\le \\sqrt{2d(d - \\lambda_2)}. \\]\n"
            "A d-regular graph is Ramanujan if \\max(|\\lambda_2|, |\\lambda_n|) \\le 2\\sqrt{d - 1}."
        ),
        "hypotheses": [
            "G is a d-regular connected graph on n vertices with d >= 3",
            "Second largest eigenvalue satisfies lambda_2 <= 2 * sqrt(d - 1)",
            "G has no self-loops and no multi-edges",
        ],
        "statement": (
            "Prove that for any d-regular Ramanujan graph G, the edge expansion constant satisfies:\n"
            "\\[ h(G) \\ge \\frac{d - 2\\sqrt{d-1}}{2}. \\]"
        ),
    },
]
