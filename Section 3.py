import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def format_joint(J):
    return np.array2string(J, formatter={"float_kind": lambda x: f"{x:0.6f}"})

def enumerate_columns(n=10):
    states = np.arange(2**n, dtype=np.uint16)
    bits = ((states[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.uint8)
    return bits

def within_score(bits):
    return np.sum(bits[:, :-1] == bits[:, 1:], axis=1)

def exact_joint_top_bottom(n=10, beta=1.0):
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
    return np.array([
        [(1-m_top)*(1-m_bot), (1-m_top)*m_bot],
        [m_top*(1-m_bot),     m_top*m_bot]
    ], dtype=float)

def gibbs_single_site(
    n=10, beta=1.0, n_samples=10000,
    burn_in=1000, thin=5, seed=1, init="random",
    scan="raster"
):

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
    degree = (ii > 0).astype(np.int8) + (ii < n-1).astype(np.int8) + \
             (jj > 0).astype(np.int8) + (jj < n-1).astype(np.int8)

    coords = [(i, j) for i in range(n) for j in range(n)]

    def neigh_sum_at(i, j):
        s = 0
        if i > 0:     s += x[i-1, j]
        if i < n-1:   s += x[i+1, j]
        if j > 0:     s += x[i, j-1]
        if j < n-1:   s += x[i, j+1]
        return s

    total_sweeps = burn_in + n_samples * thin
    samples = np.empty((n_samples, 2), dtype=np.int8)
    k = 0

    for t in range(total_sweeps):
        if scan == "raster":
            sweep_coords = coords
        elif scan == "random":
            perm = rng.permutation(n * n)
            sweep_coords = [coords[idx] for idx in perm]
        else:
            raise ValueError("scan must be 'raster' or 'random'")

  
        for (i, j) in sweep_coords:
            neigh = neigh_sum_at(i, j)
            s = 2 * neigh - int(degree[i, j])
            p1 = sigmoid(beta * s)
            x[i, j] = 1 if rng.random() < p1 else 0

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
        print(f" beta = {beta} ")

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
            J1 = gibbs_single_site(n=10, beta=beta, init="ones",
                                   n_samples=20000, burn_in=2000, thin=5, seed=1, scan="raster")
            J0 = gibbs_single_site(n=10, beta=beta, init="zeros",
                                   n_samples=20000, burn_in=2000, thin=5, seed=1, scan="raster")
            J_gibbs = 0.5 * (J0 + J1)
            print("Gibbs (single-site):\n", format_joint(J_gibbs))
        else:
            J_gibbs = gibbs_single_site(n=10, beta=beta, init="random",
                                        n_samples=20000, burn_in=2000, thin=5, seed=1, scan="raster")
            print("Gibbs (single-site):\n", format_joint(J_gibbs))
