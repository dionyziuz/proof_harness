# The Arithmetic Kakeya Conjecture

- **Problem ID**: `arithmetic-kakeya`
- **Field**: Number theory
- **Notability**: Solid result
- **Tags**: Construction - Finite, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

\section{Introduction}

A Kakeya set in $\mathbb{R}^d$ is a bounded set which contains a unit line segment in every direction. While such sets can have measure zero, the Kakeya conjecture posits that they always have the maximal possible (Minkowski and Hausdorff) dimension $d$. The arithmetic Kakeya conjecture was implicitly formulated by Katz and Tao, as a statement strong enough to imply (a form of) the Kakeya conjecture, but which can be formulated as a purely additive combinatorics statement.

One way to state the arithmetic Kakeya problem is the following: we say that $\mathrm{AK}(\alpha)$ holds if for any $\epsilon>0$ there exists a finite set $X\subset \mathbb{Z}^2$ such that $r_1+r_2\neq 0$ for all $\mathbf{x}\in X$, and for all finite $G\subset \mathbb{Z}^2$,
\[\lvert(1,-1)\cdot G\rvert\ll_\epsilon \max_{\mathbf{x}\in X} \lvert\mathbf{x}\cdot G\rvert^{\alpha+\epsilon},\]
where we write
\[\mathbf{x}\cdot G = \{ x_1g_1+x_2g_2 : (g_1,g_2)\in G\}.\]
Note that we are not interested in the size or nature of $X$ itself -- for the applications to the Kakeya conjecture itself, all that matters is the exponent $\alpha$.

It is trivial that $\mathrm{AK}(2)$ holds (for example by taking $X$ to be any two linearly independent vectors in $\mathbb{Z}^2$, as then $(1,-1)$ can be expressed as a linear combination of these two). The arithmetic Kakeya conjecture itself is the following.
\begin{conjecture}[Arithmetic Kakeya conjecture]
$\mathrm{AK}(1)$ holds.
\end{conjecture}

The relation to the Kakeya problem is the following.
\begin{theorem}
If $\mathrm{AK}(\alpha)$ holds then, for all $d\geq 2$, if $E\subseteq \mathbb{R}^d$ is a Kakeya set (a bounded set containing a unit line segment in every direction) then the Hausdorff dimension of $E$ is at least
\[\alpha^{-1}d+1-\alpha^{-1}.\]
\end{theorem}
This was implicitly established by Bourgain using the `method of slices', although Bourgain only established this for the Minkowski dimension. The stronger statement that this same bound holds for the Hausdorff dimension is a consequence of combining Bourgain's argument  with recent progress on Szemer\'{e}di's theorem by Leng, Sah, and Sawhney.

\section{Intuitive Set-up}
We will first explicitly define the kind of structure that is required, and what the `score' of such a structure (i.e. what value of $\alpha$ a structure would prove $\mathrm{AK}(\alpha)$ for). In this section we have presented things to be intuitive for the reader; an equivalent, but more opaque, version which is more suitable for verification is given in the second problem statement.

We first define a collection of constructible configurations iteratively.
\begin{definition}
Let $X\subset \mathbb{Z}^2$. An $X$-constructible graph is a (simple undirected) finite graph, in which each edge is labelled with some $\mathbf{v}\in \mathbb{Z}^2$, defined in the following iterative fashion.
\begin{enumerate}
\item A single vertex is constructible.
\item If $H$ is constructible then, for any integer $k\geq 1$, if $X_1,\ldots,X_k$ are sets of vertices of $H$, then the following graph is constructible: take $k+1$ disjoint copies of $H$ and then, for each $1\leq i\leq k$ and $x\in X_i$, either identify the vertices corresponding to $x$ in both $H_{i-1}$ and $H_i$, or put an edge between them labelled with some $\mathbf{v}\in X$.
\end{enumerate}
\end{definition}

We write $\mathbf{x}g$ for the matrix which is zero everywhere except in the column corresponding to $g$, in which it is the column vector $\mathbf{x}\in \mathbb{Z}^2$.

\begin{definition}
Let $G$ be a finite graph with vertex set $V$, such that each edge is labelled with some $\mathbf{x}\in \mathbb{Z}^2$. An $X$-\emph{forcing pair} for $G$ is a pair of sets $(R,T)$ with $T\subseteq V$ and $R$ a finite collection $2\times \left\lvert V\right\rvert$ integer matrices, with columns indexed by $g\in V$, such that
\[R\subseteq \{ \mathbf{x}g: \mathbf{x}\in X\textrm{ and }g\in V\}\]
such that a finite sequence of the following operations on $(R,T)$ eventually produces $T=V$:
\begin{enumerate}
\item If $g_1\sim g_2$ is an edge of $G$ labelled $\mathbf{v}$ then add $\mathbf{v}g_1-\mathbf{v}g_2$ to $R$.
\item If $(1,-1)g\in R$ then add $g$ to $T$.
\item Any $\mathbb{Z}$-linear combination of matrices in $R$ can be added to $R$.
\end{enumerate}

\end{definition}

\begin{definition}
Let $X\subset \mathbb{Z}^2\backslash \langle (1,-1)\rangle$ be a finite set. A constructible proof of
\[\alpha(X)\leq \frac{m+r}{n-t}\]
is an $X$-constructible graph $G$ with vertex set $V$, with $n$ vertices and $m$ edges, for which there is a forcing pair $(R,T)$ with $\left\lvert R\right\rvert=r$ and $\left\lvert T\right\rvert=t$, where $T\subseteq V$ and
\[R\subseteq \{ \mathbf{x}g : \mathbf{x}\in X\textrm{ and }g\in V\}.\]
\end{definition}

\section{Verifiable Set-up}

In this reformulation we write $d_1\times\cdots \times d_k$ for the set of lists of integers $[e_1,\ldots,e_k]$ such that $1\leq e_i\leq d_i$ for all $1\leq i\leq k$. If $X\subset \mathbb{Z}^2$ is a finite set then an $X$-constructible graph $G$ is a list of non-negative integers $d_1,...,d_k$ together with a list of functions\footnote{The function $f_i$ represents the fact that when constructing the glued graph at stage $i$ we connect the vertices corresponding to $(e_1,\ldots,e_i)$ and $(e_1,\ldots,e_i+1)$ with an edge labelled $f_i(e_1,\ldots,e_i)$.}
\[f_i: d_1\times \cdots \times d_{i-1}\times (d_i-1)\to X\]
for $1\leq i\leq k$. Let $n(G)=d_1\cdots d_k$ and
\[m(G)=\sum_{1\leq i\leq k}\sum_{\mathbf{e}}1_{f_i(\mathbf{e})\neq (0,0)}d_{i+1}\cdots d_k.\]
An $X$-set for $G$ is a pair $(R,T)$ where $T$ is a (possibly empty) list of vertices $\mathbf{e}\in d_1\times \cdots \times d_k$ and $R$ is a list of functions $d_1\times \cdots\times d_k \to X$ such that
\[\#\{ \mathbf{e}\in d_1\times \cdots\times d_k : f(\mathbf{e})\neq (0,0)\}=1\]
for all $f\in R$.

An $X$-set for $G$ is forcing if a finite number of the following operations on $(R,T)$ results in $T=d_1\times \cdots \times d_k$:
\begin{enumerate}
\item If $f_i(a_1,\ldots,a_{i})=x$ and
\[\mathbf{e}_1=(a_1,\ldots,a_{i-1},a_i,c_{i+1},\ldots,c_k)\quad\textrm{and}\quad \mathbf{e}_2=(a_1,\ldots,a_{i-1},a_i+1,c_{i+1},\ldots,c_k)\]
for some $c_{i+1},\ldots,c_k$ -- that is, $\mathbf{e}_1$ and $\mathbf{e}_2$ have the prefixes $a_1,\ldots,a_i$ and $a_1,\ldots,a_i+1$ respectively and \emph{agree in every coordinate after the $i$th} -- then add the function which takes $\mathbf{e}_1\mapsto x$ and $\mathbf{e}_2\mapsto -x$ and $\mathbf{g}\mapsto (0,0)$ for every $\mathbf{g}\neq \mathbf{e}_1,\mathbf{e}_2$ to $R$.
\item If there exists $a\neq 0$ and $\mathbf{e}$ such that $f(\mathbf{e})=(a,-a)$ and $f(\mathbf{f})=(0,0)$ for every $\mathbf{f}\not\in T\cup \{\mathbf{e}\}$ then add $\mathbf{e}$ to $T$.
\item If $f,g\in R$ then $f-g$ can be added to $R$.
\end{enumerate}

Given an $X$-constructible graph $G$ with a forcing pair $(R,T)$ its score is
\[S(X,G,R,T)=\frac{m(G)+\left\lvert R\right\rvert}{n(G)-\left\lvert T\right\rvert}.\]

\section{Problem}

Find a finite set $X\subset \mathbb{Z}^2$ such that $(0,0)\in X$ and if $(a,b)\in X\backslash \{(0,0)\}$ then $a+b\neq 0$ and an $X$-constructible graph $G$ with a forcing pair $(R,T)$ with score \leq 1.675.

Return this as a multi-line string. In the first line list the score (as a fraction) together with the parameters $m(G),\left\lvert R\right\rvert,n(G),\left\lvert T\right\rvert$. (This is not used by the verifier code, but is helpful for humans.) In the second line list the elements in $X$ as a list of pairs of integers. In the third line list the integers $d_1,\ldots,d_k$. In the fourth line list the functions $f_1,\ldots,f_k$ that make up $G$, each represented as a dictionary from $d_1\times\cdots \times (d_i-1)\to X$. These can be partial functions: omitted entries are assumed to be assigned $(0,0)$. In the fifth line list the elements of $T$. In the sixth line list the elements of $R$, each of which is a dictionary assigning an element of $X$ to some vertex of $G$.
