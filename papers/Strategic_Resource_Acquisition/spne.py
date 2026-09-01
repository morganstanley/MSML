import numpy as np

def spne_trajectories(T, alpha, beta, S, V1, V2):
    """
    Compute the subgame-perfect Nash-equilibrium (SPNE) trajectories for the
    two-player unconstrained Markovian execution game described in Algorithm 1.
    This does not implement the reserve price.

    Parameters
    ----------
    T : int
        Time horizon (number of periods, indexed 1 … T).
    alpha, beta : float
        Temporary‐ and permanent‐impact cost parameters (α, β ≥ 0, β > 0 for
        strict convexity).
    S : (T,) array-like
        Exogenous supply vector (S₁, …, S_T).  Cumulative supply enters the
        cost function.
    V1, V2 : float
        Target volumes for players 1 and 2 (terminal positions).

    Returns
    -------
    h1, h2 : ndarray, shape (T,)
        SPNE demand trajectories (h_{1,1} … h_{1,T}) and
        (h_{2,1} … h_{2,T}).
    K, L, M : ndarray, shape (2, T + 1)
        SPNE policy parameters.
    """

    S = np.asarray(S, dtype=float)
    if len(S) != T:
        raise ValueError("Supply vector S must have length T")

    # ------------------------------------------------------------------
    # 1.  Storage for backward-induction objects
    # ------------------------------------------------------------------
    # Coefficients A … F, and policy parameters K, L, M.
    A = np.zeros((2, T + 1))
    B = np.zeros((2, T + 1))
    C = np.zeros((2, T + 1))
    D = np.zeros((2, T + 1))
    E = np.zeros((2, T + 1))
    F = np.zeros((2, T + 1))
    K = np.zeros((2, T + 1))
    L = np.zeros((2, T + 1))
    M = np.zeros((2, T + 1))

    V = np.array([V1, V2], dtype=float)
    S_cum = np.cumsum(S)                       # S_{1:t} for every t

    # ------------------------------------------------------------------
    # 2.  Base step (t = T)
    # ------------------------------------------------------------------
    t = T
    for i in (0, 1):
        j = 1 - i
        sumS = S_cum[-1]

        A[i, t] = beta
        B[i, t] = 0.0
        C[i, t] = beta
        D[i, t] = alpha * (sumS - V[i] - V[j]) - beta * (2 * V[i] + V[j])
        E[i, t] = -beta * V[i]
        F[i, t] = (alpha * (V[i] ** 2 + V[i] * V[j] - V[i] * sumS) +
                   beta * (V[i] ** 2 + V[i] * V[j]))

    # ------------------------------------------------------------------
    # 3.  Backward induction for t = T-1 … 1
    # ------------------------------------------------------------------
    for t in range(T - 1, 0, -1):
        # a)  First-order (best-response) coefficients
        G = np.zeros(2)
        H = np.zeros(2)
        Icoef = np.zeros(2)
        Jcoef = np.zeros(2)

        for i in (0, 1):
            denom = 2.0 * (alpha + beta + A[i, t + 1])
            G[i] = -1 * (alpha + beta + C[i, t + 1]) / denom
            H[i] = -1 * (alpha + 2.0 * A[i, t + 1]) / denom
            Icoef[i] = -1 * (alpha + C[i, t + 1]) / denom
            Jcoef[i] = -1 * (D[i, t + 1] - alpha * S_cum[t - 1]) / denom

        # b)  Linear policies  (simultaneous move ⇒ solve 2×2 system)
        denom_KLM = 1.0 - G[0] * G[1]

        # player 1
        K[0, t] = (G[0] * H[1] + Icoef[0]) / denom_KLM
        L[0, t] = (G[0] * Icoef[1] + H[0]) / denom_KLM
        M[0, t] = (G[0] * Jcoef[1] + Jcoef[0]) / denom_KLM
        # player 2
        K[1, t] = (G[1] * H[0] + Icoef[1]) / denom_KLM
        L[1, t] = (G[1] * Icoef[0] + H[1]) / denom_KLM
        M[1, t] = (G[1] * Jcoef[0] + Jcoef[1]) / denom_KLM

        # c)  Update quadratic-cost coefficients A … F
        for i in (0, 1):
            j = 1 - i
            Li, Ki, Mi = L[i, t], K[i, t], M[i, t]
            Lj, Kj, Mj = L[j, t], K[j, t], M[j, t]

            Aip1, Bip1, Cip1 = A[i, t + 1], B[i, t + 1], C[i, t + 1]
            Dip1, Eip1 = D[i, t + 1], E[i, t + 1]
            sumS = S_cum[t - 1]

            #   A_i^t
            A[i, t] = (Li ** 2) * (alpha + beta + Aip1) \
                      + Li * (alpha + 2.0 * Aip1) \
                      + Li * Kj * (alpha + beta + Cip1) \
                      + Aip1 + Kj * Cip1 + (Kj ** 2) * Bip1

            #   B_i^t
            B[i, t] = (Ki ** 2) * (alpha + beta + Aip1) \
                      + Ki * (alpha + Cip1) \
                      + Lj * Ki * (alpha + beta + Cip1) \
                      + Bip1 * (1.0 + 2.0 * Lj + Lj ** 2)

            #   C_i^t
            C[i, t] = (2.0 * Ki * Li) * (alpha + beta + Aip1) \
                      + Ki * (alpha + 2.0 * Aip1) \
                      + Li * (alpha + Cip1) \
                      + (Li * Lj + Ki * Kj) * (alpha + beta + Cip1) \
                      + Cip1 + Lj * Cip1 \
                      + 2.0 * Kj * Bip1 + 2.0 * Kj * Lj * Bip1

            #   D_i^t
            D[i, t] = (2.0 * Li * Mi) * (alpha + beta + Aip1) \
                      + Mi * (alpha + 2.0 * Aip1) \
                      + (Mi * Kj + Li * Mj) * (alpha + beta + Cip1) \
                      + Li * (Dip1 - alpha * sumS) \
                      + Mj * Cip1 \
                      + Dip1 + Kj * Eip1 + 2.0 * Kj * Mj * Bip1

            #   E_i^t  ← corrected (1 + L_{-i}^t) factor
            E[i, t] = (2.0 * Ki * Mi) * (alpha + beta + Aip1) \
                      + Mi * (alpha + Cip1) \
                      + (Mi * Lj + Ki * Mj) * (alpha + beta + Cip1) \
                      + Ki * (Dip1 - alpha * sumS) \
                      + (1.0 + Lj) * Eip1 \
                      + 2.0 * (1.0 + Lj) * Bip1 * Mj

            #   F_i^t
            F[i, t] = (Mi ** 2) * (alpha + beta + Aip1) \
                      + Mi * Mj * (alpha + beta + Cip1) \
                      + Mi * (Dip1 - alpha * sumS) \
                      + Mj * Eip1 \
                      + (Mj ** 2) * Bip1 \
                      + F[i, t + 1]

    # ------------------------------------------------------------------
    # 4.  Forward simulation of the realised trajectories
    # ------------------------------------------------------------------
    h1 = np.zeros(T)
    h2 = np.zeros(T)
    q1 = q2 = 0.0

    for t in range(1, T):
        h1_t = K[0, t] * q2 + L[0, t] * q1 + M[0, t]
        h2_t = K[1, t] * q1 + L[1, t] * q2 + M[1, t]

        h1[t - 1] = h1_t
        h2[t - 1] = h2_t

        q1 += h1_t
        q2 += h2_t

    # Last period: buy remaining inventory exactly
    h1[-1] = V1 - q1
    h2[-1] = V2 - q2

    # return trajectories and policy parameters
    return h1, h2, K, L, M


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    T = 3
    alpha = 1 # permanent impact
    beta = 1 # temporary impact
    S = np.zeros(T)
    V1, V2 = 5, 5

    h1, h2, K, L, M = spne_trajectories(T, alpha, beta, S, V1, V2)
    print(f"T={T}, alpha={alpha}, beta={beta}, S={S}, V1={V1}, V2={V2}")
    print("Player 1 trajectory:", h1)
    print("Player 2 trajectory:", h2)
    print("Final positions:", np.sum(h1), np.sum(h2))
