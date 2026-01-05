import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def format_joint(J):
    return np.array2string(J, formatter={"float_kind": lambda x: f"{x:0.6f}"})
def enumerate_columns(n=10):
    """All 2^n column states as 0/1 bit-vectors, top bit first."""
    states = np.arange(2**n, dtype=np.uint16)
    bits = ((states[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.uint8)
    return bits

def within_score(bits):
    """Vertical agreements count in a column state (n-1 adjacent pairs)."""
    return np.sum(bits[:, :-1] == bits[:, 1:], axis=1)

def exact_joint_top_bottom(n=10, beta=1.0):
    """
    Computes exact P(x_{1,n}, x_{n,n}) by:
      - cluster each column into X_t (2^n states)
      - chain forward recursion (sum-product on induced chain)
      - aggregate over last-column states matching (top,bottom).
    """
    bits = enumerate_columns(n)
    S = 2**n

    vscore = within_score(bits)
    node_pot = np.exp(beta * vscore)

    hscore = (bits[:, None, :] == bits[None, :, :]).sum(axis=2)
    edge_pot = np.exp(beta * hscore)

    alpha = node_pot.copy()
    alpha /= alpha.sum()
    for _ in range(1, n):
        alpha = node_pot * (alpha @ edge_pot)
        alpha /= alpha.sum()

    p_last = alpha / alpha.sum()

    top = bits[:, 0]
    bottom = bits[:, -1]
    joint = np.zeros((2, 2), dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            joint[a, b] = p_last[(top == a) & (bottom == b)].sum()
    return joint


def mean_field_coordinate_ascent(
    n=10, beta=1.0, max_sweeps=5000, tol=1e-10,
    seed=1, init="random", damping=0.0, order="raster"
):
    """
    Fully-factorised mean field with *coordinate ascent* (Gauss–Seidel / CAVI).

    Variational family:
        q(x) = Π_{i,j} Bernoulli(m_{ij}),   m_{ij} = q_{ij}(x_{ij}=1).

    Coordinate update for site (i,j):
        m_{ij} <- sigmoid( beta * sum_{nbr}(2*m_nbr - 1) )

    Update schedule:
      - order="raster": deterministic scan i=0..n-1, j=0..n-1 each sweep
      - order="random": random permutation of sites each sweep

    Damping:
        m <- (1-damping)*new + damping*old
    """
    rng = np.random.default_rng(seed)

    if init == "random":
        m = rng.uniform(0.25, 0.75, size=(n, n))
    elif init == "half":
        m = np.full((n, n), 0.5)
    else:
        raise ValueError("init must be 'random' or 'half'")

    idx = [(i, j) for i in range(n) for j in range(n)]

    for _ in range(max_sweeps):
        if order == "random":
            rng.shuffle(idx)
        elif order != "raster":
            raise ValueError("order must be 'raster' or 'random'")

        max_delta = 0.0

        for (i, j) in idx:
            field = 0.0
            if i > 0:     field += (2.0 * m[i-1, j] - 1.0)
            if i < n - 1: field += (2.0 * m[i+1, j] - 1.0)
            if j > 0:     field += (2.0 * m[i, j-1] - 1.0)
            if j < n - 1: field += (2.0 * m[i, j+1] - 1.0)

            new = sigmoid(beta * field)

            if damping > 0.0:
                new = (1.0 - damping) * new + damping * m[i, j]

            delta = abs(new - m[i, j])
            if delta > max_delta:
                max_delta = delta

            m[i, j] = new

        if max_delta < tol:
            break

    return m

def mf_joint_from_m(m_top, m_bot):
    """Because MF factorises: q(top,bottom) = q(top) q(bottom)."""
    return np.array([
        [(1-m_top)*(1-m_bot), (1-m_top)*m_bot],
        [m_top*(1-m_bot),     m_top*m_bot]
    ], dtype=float)


def gibbs_checkerboard(n=10, beta=1.0, n_samples=20000,
                       burn_in=2000, thin=5, seed=1, init="random"):
    """
    Vectorised checkerboard Gibbs:
      - update all black sites given white
      - update all white sites given black
    Returns joint estimate for (top,bottom) of rightmost column.
    """
    rng = np.random.default_rng(seed)
    if init == "random":
        x = rng.integers(0, 2, size=(n, n), dtype=np.int8)
    elif init == "zeros":
        x = np.zeros((n, n), dtype=np.int8)
    elif init == "ones":
        x = np.ones((n, n), dtype=np.int8)
    else:
        raise ValueError("init must be random/zeros/ones")

    ii, jj = np.indices((n, n))
    black = ((ii + jj) % 2 == 0)
    white = ~black

    degree = (ii > 0).astype(np.int8) + (ii < n-1).astype(np.int8) + \
             (jj > 0).astype(np.int8) + (jj < n-1).astype(np.int8)

    def sum_neighbors(x):
        s = np.zeros_like(x, dtype=np.int16)
        s[1:, :]  += x[:-1, :]
        s[:-1, :] += x[1:, :]
        s[:, 1:]  += x[:, :-1]
        s[:, :-1] += x[:, 1:]
        return s

    total_sweeps = burn_in + n_samples * thin
    samples = np.empty((n_samples, 2), dtype=np.int8)
    k = 0

    for t in range(total_sweeps):
        neigh = sum_neighbors(x)
        s = 2 * neigh - degree
        p1 = sigmoid(beta * s)
        r = rng.random(size=(n, n))
        x[black] = (r[black] < p1[black]).astype(np.int8)

        neigh = sum_neighbors(x)
        s = 2 * neigh - degree
        p1 = sigmoid(beta * s)
        r = rng.random(size=(n, n))
        x[white] = (r[white] < p1[white]).astype(np.int8)

        if t >= burn_in and ((t - burn_in) % thin == 0):
            samples[k, 0] = x[0, n-1]
            samples[k, 1] = x[n-1, n-1]
            k += 1

    joint = np.zeros((2, 2), dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            joint[a, b] = np.mean((samples[:, 0] == a) & (samples[:, 1] == b))
    return joint


if __name__ == "__main__":
    for beta in [4, 1, 0.01]:
        print(f"\n===== beta = {beta} =====")

        J_exact = exact_joint_top_bottom(n=10, beta=beta)
        print("Exact:\n", format_joint(J_exact))

        m = mean_field_coordinate_ascent(
            n=10, beta=beta,
            seed=1, init="random",
            order="raster",
            damping=0.0,
            max_sweeps=5000, tol=1e-10
        )
        J_mf = mf_joint_from_m(m[0, -1], m[-1, -1])
        print("Mean Field (coord ascent):\n", format_joint(J_mf))

        if beta == 4:
            J1 = gibbs_checkerboard(n=10, beta=beta, init="ones")
            J0 = gibbs_checkerboard(n=10, beta=beta, init="zeros")
            J_gibbs = 0.5 * (J0 + J1)
            print("Gibbs:\n", format_joint(J_gibbs))
        else:
            J_gibbs = gibbs_checkerboard(n=10, beta=beta, init="random")
            print("Gibbs:\n", format_joint(J_gibbs))
