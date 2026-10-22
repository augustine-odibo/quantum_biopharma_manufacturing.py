#!/usr/bin/env python3
"""
Quantum-Enhanced Biopharmaceutical Manufacturing Engine
========================================================

A runnable reference implementation of how quantum algorithms plug into the
MANUFACTURING phase of biologics production (upstream cell culture, downstream
purification, fill/finish), wired together as one closed control loop:

    ┌──────────────────────┐   risk per    ┌───────────────────────┐   schedule   ┌────────────────┐
    │ 1. Quantum-kernel PAT│ ───reactor──▶ │ 2. QAOA batch-to-     │ ───────────▶ │ Execution / MES│
    │    quality predictor │               │    bioreactor schedule│              │  (Shop floor)  │
    └──────────────────────┘               └───────────────────────┘              └────────────────┘
              ▲                                                                            │
              │                    sensor data (pH, DO, temp, feed)                        │
              └────────────────────────────────────────────────────────────────────────────┘
    ┌──────────────────────────────────────────────┐
    │ 3. VQE molecular energy engine (process       │  feeds reaction/stability parameters
    │    chemistry: catalysts, excipients, linkers) │  into the process model
    └──────────────────────────────────────────────┘

Module 1  Quantum kernel classifier (ZZ feature map) trained on Process
          Analytical Technology (PAT) sensor data to predict out-of-spec
          critical quality attributes (CQAs) before the batch fails.
Module 2  QAOA with a CVaR objective solving a QUBO that assigns batches to
          bioreactors, honouring time conflicts, cleaning/changeover cost,
          scale fit, and the live quality risk produced by Module 1.
Module 3  VQE computing the ground-state energy of a molecular Hamiltonian
          (H2 benchmark), the primitive behind quantum chemistry for process
          chemistry, catalyst screening and formulation stability.

Everything runs on an exact statevector simulator written in NumPy, so the
code is fully reproducible on a laptop. The QUBO and circuits are hardware
agnostic: the QUBO matrix can be sent unchanged to a quantum annealer, and the
circuits map gate-for-gate onto Qiskit / Cirq / PennyLane for real hardware.

Honest scope: problem sizes here are small enough to verify classically, which
is the point; it lets every quantum result be checked against ground truth.
Practical quantum advantage for problems of plant scale has not yet been
demonstrated; this code is the architecture you would scale as hardware
matures, and the benchmarks tell you, objectively, when it beats classical.

Requirements: Python 3.9+, numpy, scipy.
Run:          python quantum_biopharma_manufacturing.py
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

RNG_SEED = 7


# =============================================================================
# 0. Exact statevector simulator (qubit 0 = most significant bit)
# =============================================================================
class Statevector:
    """Minimal, exact n-qubit simulator. Gates act via tensor contraction."""

    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    @staticmethod
    def ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def zero(n: int) -> np.ndarray:
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0
        return psi

    @staticmethod
    def apply_1q(psi: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
        t = psi.reshape([2] * n)
        t = np.tensordot(U, t, axes=([1], [q]))
        return np.moveaxis(t, 0, q).reshape(-1)

    @staticmethod
    def apply_2q(psi: np.ndarray, U: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
        t = psi.reshape([2] * n)
        t = np.tensordot(U.reshape(2, 2, 2, 2), t, axes=([2, 3], [q1, q2]))
        return np.moveaxis(t, [0, 1], [q1, q2]).reshape(-1)

    @staticmethod
    def bit_table(n: int) -> np.ndarray:
        """bits[k, q] = value of qubit q in computational basis state k."""
        k = np.arange(2**n)[:, None]
        return ((k >> (n - 1 - np.arange(n))) & 1).astype(np.int8)


SV = Statevector


# =============================================================================
# 1. QUANTUM-KERNEL PAT QUALITY PREDICTOR
# =============================================================================
@dataclass
class PATDataset:
    """Synthetic Process Analytical Technology data for fed-batch mAb runs.

    Features (per batch, day-7 snapshot): pH deviation, dissolved O2 (%),
    temperature deviation (°C), glucose feed rate (g/L/day).
    Label: +1 = CQA out of spec (e.g. high-mannose glycoform > limit), -1 = in spec.
    The ground-truth risk surface is deliberately non-linear (interaction terms),
    mimicking how glycosylation responds jointly to pH, temperature and feed.
    """

    X_raw: np.ndarray
    y: np.ndarray
    feature_names: tuple = ("pH_dev", "DO_pct", "temp_dev_C", "glucose_feed")

    @classmethod
    def generate(cls, n: int, rng: np.random.Generator) -> "PATDataset":
        ph = rng.normal(0.0, 0.08, n)
        do = rng.normal(40.0, 8.0, n)
        temp = rng.normal(0.0, 0.6, n)
        glc = rng.normal(4.0, 1.0, n)
        risk = (
            9.0 * ph * temp
            + 0.9 * np.sin(0.25 * (do - 40.0)) * (glc - 4.0)
            + 0.6 * (temp**2)
            - 0.5
            + rng.normal(0, 0.15, n)
        )
        y = np.where(risk > 0, 1.0, -1.0)
        return cls(np.column_stack([ph, do, temp, glc]), y)


class QuantumKernelClassifier:
    """Kernel ridge classifier with a ZZ-feature-map fidelity kernel.

    |φ(x)> = U_Φ(x) H^n U_Φ(x) H^n |0>,  U_Φ(x) = exp(i[Σ x_i Z_i + Σ (π-x_i)(π-x_j) Z_i Z_j])
    k(x, x') = |<φ(x)|φ(x')>|²  -- estimated on hardware with a swap / compute-uncompute test.
    """

    def __init__(self, n_features: int, reps: int = 2, bandwidth: float = 1.0, reg: float = 1e-2):
        self.n = n_features
        self.reps = reps
        self.bandwidth = bandwidth  # scaling against kernel concentration
        self.reg = reg
        bits = SV.bit_table(self.n)
        self.spins = 1 - 2 * bits.astype(float)  # Z eigenvalues ±1
        Hn = np.array([[1.0]])
        for _ in range(self.n):
            Hn = np.kron(Hn, SV.H.real)
        self.Hn = Hn
        self.pairs = list(itertools.combinations(range(self.n), 2))

    # --- preprocessing: scale each feature to [0, π] using training statistics
    def _fit_scaler(self, X: np.ndarray) -> None:
        self.lo, self.hi = X.min(0), X.max(0)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.lo) / (self.hi - self.lo + 1e-12), 0, 1) * np.pi * self.bandwidth

    def feature_states(self, X: np.ndarray) -> np.ndarray:
        """Batch-simulate |φ(x)> for every row of X. Returns (m, 2^n)."""
        m = X.shape[0]
        phase = X @ self.spins.T  # Σ x_i s_i  -> (m, 2^n)
        for i, j in self.pairs:
            phase += np.outer((np.pi - X[:, i]) * (np.pi - X[:, j]), self.spins[:, i] * self.spins[:, j])
        D = np.exp(1j * phase)
        psi = np.zeros((m, 2**self.n), dtype=complex)
        psi[:, 0] = 1.0
        for _ in range(self.reps):
            psi = (psi @ self.Hn.T) * D
        return psi

    def kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.abs(A.conj() @ B.T) ** 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelClassifier":
        self._fit_scaler(X)
        self.S_train = self.feature_states(self._scale(X))
        K = self.kernel(self.S_train, self.S_train)
        self.alpha = np.linalg.solve(K + self.reg * np.eye(len(y)), y)
        self.K_offdiag_var = float(np.var(K[~np.eye(len(K), dtype=bool)]))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        S = self.feature_states(self._scale(X))
        return self.kernel(S, self.S_train) @ self.alpha

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelClassifier":
        """Platt scaling: fit P(y=+1|f) = σ(a·f + b) on held-out decision values."""
        f, t = self.decision_function(X), (y > 0).astype(float)

        def nll(ab):
            z = ab[0] * f + ab[1]
            return float(np.sum(np.logaddexp(0, z) - t * z))

        self.platt = minimize(nll, np.array([1.0, 0.0]), method="BFGS").x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability that the batch's CQA goes out of spec."""
        a, b = getattr(self, "platt", (3.0, 0.0))
        return 1.0 / (1.0 + np.exp(-(a * self.decision_function(X) + b)))


class RBFKernelBaseline:
    """Classical comparator: same kernel-ridge machinery, Gaussian kernel."""

    def __init__(self, gamma: float = 0.5, reg: float = 1e-2):
        self.gamma, self.reg = gamma, reg

    def _z(self, X):
        return (X - self.mu) / self.sd

    def _k(self, A, B):
        d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-self.gamma * d2)

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.Xt = self._z(X)
        self.alpha = np.linalg.solve(self._k(self.Xt, self.Xt) + self.reg * np.eye(len(y)), y)
        return self

    def decision_function(self, X):
        return self._k(self._z(X), self.Xt) @ self.alpha


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC, no external deps."""
    pos, neg = score[y_true > 0], score[y_true < 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tune_quantum_kernel(X, y, rng, bandwidths=(0.05, 0.1, 0.15, 0.2, 0.3), regs=(1e-3, 1e-2, 1e-1)):
    """3-fold CV over feature-map bandwidth and ridge strength.

    Bandwidth is the key lever against exponential kernel concentration: at
    full-angle encoding the fidelity kernel flattens toward a constant and the
    model cannot generalise (Thanasilp et al., 2022; Shaydulin & Wild, 2022).
    """
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 3)
    best, best_auc = (bandwidths[0], regs[0]), -1.0
    for bw, reg in itertools.product(bandwidths, regs):
        aucs = []
        for k in range(3):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(3) if j != k])
            m = QuantumKernelClassifier(X.shape[1], bandwidth=bw, reg=reg).fit(X[tr], y[tr])
            aucs.append(roc_auc(y[va], m.decision_function(X[va])))
        if np.mean(aucs) > best_auc:
            best, best_auc = (bw, reg), float(np.mean(aucs))
    return best, best_auc


# =============================================================================
# 2. QAOA BATCH-TO-BIOREACTOR SCHEDULER (QUBO)
# =============================================================================
@dataclass
class SchedulingProblem:
    batches: list
    reactors: list
    base_cost: np.ndarray          # (B, R) changeover + cleaning + scale-fit cost
    conflicts: list                # (b1, b2) pairs overlapping in time -> cannot share a reactor
    reactor_risk: np.ndarray       # (R,) P(out-of-spec) from the quantum PAT model
    risk_weight: float = 6.0
    penalty: float = 20.0
    Q: np.ndarray = field(init=False)
    offset: float = field(init=False)

    def __post_init__(self):
        self.B, self.R = self.base_cost.shape
        self.n = self.B * self.R
        self.Q, self.offset = self.build_qubo()

    def var(self, b: int, r: int) -> int:
        return b * self.R + r

    def build_qubo(self):
        """Energy(x) = x^T Q x + offset, with x ∈ {0,1}^(B·R), x[b,r]=1 ⇔ batch b on reactor r."""
        Q = np.zeros((self.n, self.n))
        offset = 0.0
        P = self.penalty
        cost = self.base_cost + self.risk_weight * self.reactor_risk[None, :]
        for b in range(self.B):
            for r in range(self.R):
                Q[self.var(b, r), self.var(b, r)] += cost[b, r]
            # one-hot: P * (1 - Σ_r x_br)^2 = P*(1 - Σx + 2Σ_{r<r'} x x')   (x² = x)
            offset += P
            for r in range(self.R):
                Q[self.var(b, r), self.var(b, r)] -= P
            for r1, r2 in itertools.combinations(range(self.R), 2):
                Q[self.var(b, r1), self.var(b, r2)] += P
                Q[self.var(b, r2), self.var(b, r1)] += P
        for b1, b2 in self.conflicts:  # time-overlap exclusion
            for r in range(self.R):
                Q[self.var(b1, r), self.var(b2, r)] += P / 2
                Q[self.var(b2, r), self.var(b1, r)] += P / 2
        return Q, offset

    def energies(self) -> np.ndarray:
        X = SV.bit_table(self.n).astype(float)
        return np.einsum("ki,ij,kj->k", X, self.Q, X) + self.offset

    def decode(self, k: int) -> dict:
        bits = SV.bit_table(self.n)[k].reshape(self.B, self.R)
        return {self.batches[b]: [self.reactors[r] for r in np.flatnonzero(bits[b])] for b in range(self.B)}

    def is_feasible(self, k: int) -> bool:
        bits = SV.bit_table(self.n)[k].reshape(self.B, self.R)
        if not np.all(bits.sum(1) == 1):
            return False
        return all(np.argmax(bits[b1]) != np.argmax(bits[b2]) for b1, b2 in self.conflicts)


class QAOAScheduler:
    """QAOA (depth p) with a CVaR_α objective (Barkoutsos et al., Quantum 2020).

    mixer="x"  : standard transverse-field mixer from |+>^n; one-hot rules are
                 enforced only by QUBO penalties, so amplitude leaks to
                 infeasible schedules.
    mixer="xy" : Quantum Alternating Operator Ansatz (Hadfield et al., 2019).
                 Start in a W-state per batch and mix with ring XY (hopping)
                 terms, which preserve Hamming weight, so every sampled
                 schedule assigns each batch to exactly one reactor.
    CVaR focuses the optimiser on the best α-fraction of measured schedules,
    raising the probability of sampling the true optimum vs. plain <H>.
    """

    def __init__(self, problem: SchedulingProblem, p: int = 3, alpha: float = 0.1,
                 restarts: int = 8, mixer: str = "xy"):
        self.pb, self.p, self.alpha, self.restarts, self.mixer = problem, p, alpha, restarts, mixer
        self.E = problem.energies()
        feas = np.array([problem.is_feasible(k) for k in range(len(self.E))])
        ref = self.E[feas] if mixer == "xy" else self.E
        self.C = (self.E - ref.min()) / (np.ptp(ref) + 1e-12)  # normalised diagonal cost Hamiltonian
        self.order = np.argsort(self.E)
        self.n_evals = 0
        R = problem.R
        self.ring = [(problem.var(b, r), problem.var(b, (r + 1) % R))
                     for b in range(problem.B) for r in range(R if R > 2 else 1)]
        self.psi0 = self._initial_state()

    def _initial_state(self) -> np.ndarray:
        n = self.pb.n
        if self.mixer == "x":
            return np.full(2**n, 1 / np.sqrt(2**n), dtype=complex)
        w = np.zeros(2**self.pb.R, dtype=complex)  # W-state on each batch register
        for r in range(self.pb.R):
            w[1 << (self.pb.R - 1 - r)] = 1 / np.sqrt(self.pb.R)
        psi = np.array([1.0 + 0j])
        for _ in range(self.pb.B):
            psi = np.kron(psi, w)
        return psi

    @staticmethod
    def xy_gate(beta: float) -> np.ndarray:
        """exp(-iβ(XX+YY)/2): swaps excitation between two qubits, conserves weight."""
        c, s = np.cos(beta), np.sin(beta)
        return np.array([[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, 1]], dtype=complex)

    def state(self, params: np.ndarray) -> np.ndarray:
        n = self.pb.n
        gammas, betas = params[: self.p], params[self.p :]
        psi = self.psi0.copy()
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.C)               # e^{-iγ H_C}
            if self.mixer == "x":
                Ux = SV.rx(2 * b)
                for q in range(n):                             # e^{-iβ Σ X}
                    psi = SV.apply_1q(psi, Ux, q, n)
            else:
                Uxy = self.xy_gate(b)
                for q1, q2 in self.ring:                       # Trotterised ring-XY mixer
                    psi = SV.apply_2q(psi, Uxy, q1, q2, n)
        return psi

    def cvar(self, probs: np.ndarray) -> float:
        p_sorted = probs[self.order]
        cum = np.cumsum(p_sorted)
        cut = min(int(np.searchsorted(cum, self.alpha)) + 1, len(cum))
        w = p_sorted[:cut].copy()
        w[-1] -= cum[cut - 1] - self.alpha if cum[cut - 1] > self.alpha else 0.0
        return float(np.dot(w, self.E[self.order][:cut]) / self.alpha)

    def objective(self, params: np.ndarray) -> float:
        self.n_evals += 1
        return self.cvar(np.abs(self.state(params)) ** 2)

    def solve(self, rng: np.random.Generator) -> dict:
        best = None
        for _ in range(self.restarts):
            x0 = np.concatenate([rng.uniform(0, 2 * np.pi, self.p), rng.uniform(0, np.pi, self.p)])
            # layer-wise warm start would go here for larger p
            res = minimize(self.objective, x0, method="COBYLA", options={"maxiter": 400, "rhobeg": 0.4})
            if best is None or res.fun < best.fun:
                best = res
        probs = np.abs(self.state(best.x)) ** 2
        k_opt = int(self.order[0])
        k_mode = int(np.argmax(probs))
        feasible_mass = float(sum(probs[k] for k in range(len(probs)) if self.pb.is_feasible(k)))
        return {
            "params": best.x,
            "cvar": best.fun,
            "p_optimal": float(probs[k_opt]),
            "p_feasible": feasible_mass,
            "random_guess_p_optimal": 1 / 2**self.pb.n,
            "most_likely": k_mode,
            "optimal": k_opt,
            "optimal_energy": float(self.E[k_opt]),
            "most_likely_energy": float(self.E[k_mode]),
            "shots_to_99pct": int(np.ceil(np.log(0.01) / np.log(1 - probs[k_opt]))) if probs[k_opt] < 1 else 1,
            "circuit_evals": self.n_evals,
        }


# =============================================================================
# 3. VQE MOLECULAR ENERGY ENGINE (process chemistry primitive)
# =============================================================================
class VQEMolecularEngine:
    """Hardware-efficient VQE for a qubit-mapped molecular Hamiltonian.

    Benchmark: H2 / STO-3G at 0.735 Å, parity-mapped with two-qubit reduction
    (standard coefficients). Same pipeline scales to active spaces of catalysts,
    linker chemistry for ADCs, or excipient–protein interactions as qubit
    counts and error rates allow.
    """

    H2_TERMS = {  # Pauli string -> coefficient (Hartree)
        "II": -1.052373245772859,
        "ZI": 0.39793742484318045,
        "IZ": -0.39793742484318045,
        "ZZ": -0.01128010425623538,
        "XX": 0.18093119978423156,
    }
    NUCLEAR_REPULSION = 0.7199689944489797

    def __init__(self, terms: dict | None = None, layers: int = 2):
        self.terms = terms or self.H2_TERMS
        self.n = len(next(iter(self.terms)))
        self.layers = layers
        P = {"I": SV.I2, "X": SV.X, "Z": SV.Z}
        self.Hmat = sum(c * self._kron([P[s] for s in pstr]) for pstr, c in self.terms.items())

    @staticmethod
    def _kron(mats):
        out = np.array([[1.0 + 0j]])
        for m in mats:
            out = np.kron(out, m)
        return out

    def ansatz(self, theta: np.ndarray) -> np.ndarray:
        n, psi, k = self.n, SV.zero(self.n), 0
        for _ in range(self.layers + 1):
            for q in range(n):
                psi = SV.apply_1q(psi, SV.ry(theta[k]), q, n)
                k += 1
            if _ < self.layers:
                for q in range(n - 1):
                    psi = SV.apply_2q(psi, SV.CNOT, q, q + 1, n)
        return psi

    def n_params(self) -> int:
        return self.n * (self.layers + 1)

    def energy(self, theta: np.ndarray) -> float:
        psi = self.ansatz(theta)
        return float(np.real(psi.conj() @ self.Hmat @ psi))

    def solve(self, rng: np.random.Generator, restarts: int = 5) -> dict:
        best = min(
            (minimize(self.energy, rng.uniform(-np.pi, np.pi, self.n_params()), method="BFGS")
             for _ in range(restarts)),
            key=lambda r: r.fun,
        )
        exact = float(np.linalg.eigvalsh(self.Hmat)[0])
        return {
            "vqe_electronic": best.fun,
            "exact_electronic": exact,
            "vqe_total": best.fun + self.NUCLEAR_REPULSION,
            "exact_total": exact + self.NUCLEAR_REPULSION,
            "error_mHa": abs(best.fun - exact) * 1000,
            "chemical_accuracy": abs(best.fun - exact) < 1.6e-3,
        }


# =============================================================================
# 4. ORCHESTRATION: the closed manufacturing loop
# =============================================================================
def banner(t: str) -> None:
    print("\n" + "=" * 78 + f"\n {t}\n" + "=" * 78)


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    report = {}
    t0 = time.time()

    # ---------- Module 1: quantum PAT quality prediction ----------
    banner("MODULE 1 | Quantum-kernel PAT model: predict out-of-spec CQAs early")
    data = PATDataset.generate(260, rng)
    idx = rng.permutation(len(data.y))
    tr, te = idx[:180], idx[180:]
    Xtr, ytr, Xte, yte = data.X_raw[tr], data.y[tr], data.X_raw[te], data.y[te]

    (bw, reg), cv_auc = tune_quantum_kernel(Xtr, ytr, rng)
    fit_i, cal_i = np.arange(140), np.arange(140, 180)  # hold out 40 for probability calibration
    qk = QuantumKernelClassifier(4, reps=2, bandwidth=bw, reg=reg).fit(Xtr[fit_i], ytr[fit_i])
    qk.calibrate(Xtr[cal_i], ytr[cal_i])
    q_dec = qk.decision_function(Xte)

    # Fair classical comparator: same split, gamma chosen on the same calibration slice
    rbf = max((RBFKernelBaseline(gamma=g).fit(Xtr[fit_i], ytr[fit_i]) for g in (0.05, 0.1, 0.3, 0.5, 1.0)),
              key=lambda m: roc_auc(ytr[cal_i], m.decision_function(Xtr[cal_i])))
    c_dec = rbf.decision_function(Xte)

    q_acc, c_acc = np.mean(np.sign(q_dec) == yte), np.mean(np.sign(c_dec) == yte)
    q_auc, c_auc = roc_auc(yte, q_dec), roc_auc(yte, c_dec)
    print(f"  Batches: {len(fit_i)} fit / {len(cal_i)} calibrate / {len(te)} test   "
          f"out-of-spec rate: {np.mean(data.y > 0):.1%}")
    print(f"  Feature map: ZZ, 4 qubits, 2 reps, CV-tuned bandwidth = {bw}, ridge = {reg} (CV AUC {cv_auc:.3f})")
    print(f"  Kernel off-diagonal variance (concentration check): {qk.K_offdiag_var:.4f}")
    print(f"  {'Model':<28}{'Accuracy':>10}{'ROC-AUC':>10}")
    print(f"  {'Quantum kernel (ZZ map)':<28}{q_acc:>10.3f}{q_auc:>10.3f}")
    print(f"  {'Classical RBF baseline':<28}{c_acc:>10.3f}{c_auc:>10.3f}")
    report["quality_model"] = {"bandwidth": bw, "quantum": {"acc": q_acc, "auc": q_auc},
                               "classical_rbf": {"acc": c_acc, "auc": c_auc}}

    # Live sensor snapshot of each bioreactor's current run -> risk for scheduler
    reactors = ["SUB-2000L-A", "SS-2000L-B", "SUB-2000L-C"]
    live = np.array([
        [0.02, 41.0, 0.1, 4.1],    # healthy
        [0.13, 30.0, 0.9, 5.6],    # drifting: pH & temp excursion, low DO
        [-0.03, 44.0, -0.2, 3.8],  # healthy
    ])
    risk = qk.predict_proba(live)
    print("\n  Live reactor risk (P out-of-spec) handed to scheduler:")
    for r, p in zip(reactors, risk):
        status = "CRITICAL: hold & investigate" if p > 0.5 else "ELEVATED: reroute + tighten control" if p > 0.25 else "ok"
        print(f"    {r:<14} {p:6.1%}  {status}")
    report["reactor_risk"] = dict(zip(reactors, risk.round(4).tolist()))

    # ---------- Module 2: QAOA schedule ----------
    banner("MODULE 2 | QAOA (CVaR) batch-to-bioreactor scheduling on a QUBO")
    batches = ["mAb-A", "mAb-B", "ADC-intermediate"]
    base_cost = np.array([   # changeover/cleaning + media/scale fit (arbitrary cost units)
        [2.0, 3.0, 4.0],
        [3.0, 2.5, 4.5],
        [5.0, 5.0, 1.5],
    ])
    conflicts = [(0, 1)]  # mAb-A and mAb-B overlap in time; others can run back-to-back
    pb = SchedulingProblem(batches, reactors, base_cost, conflicts, risk)
    baseline = QAOAScheduler(pb, p=4, alpha=0.5, restarts=8, mixer="x").solve(rng)
    qaoa = QAOAScheduler(pb, p=4, alpha=0.5, restarts=8, mixer="xy")
    res = qaoa.solve(rng)

    print(f"  Qubits: {pb.n}   search space: {2**pb.n} schedules   depth p = {qaoa.p}   CVaR α = {qaoa.alpha}")
    print(f"  {'Ansatz':<34}{'P(optimal)':>11}{'P(feasible)':>13}")
    print(f"  {'Standard QAOA (X-mixer)':<34}{baseline['p_optimal']:>11.3f}{baseline['p_feasible']:>13.3f}")
    print(f"  {'Constraint-preserving (XY-mixer)':<34}{res['p_optimal']:>11.3f}{res['p_feasible']:>13.3f}")
    print(f"  P(optimal schedule) after QAOA: {res['p_optimal']:.3f}  "
          f"(uniform random: {res['random_guess_p_optimal']:.4f}, "
          f"{res['p_optimal'] / res['random_guess_p_optimal']:.0f}x amplification)")
    print(f"  Probability mass on feasible schedules: {res['p_feasible']:.3f}")
    print(f"  Shots for 99% chance of sampling optimum: {res['shots_to_99pct']}")
    print(f"  Most likely sample == brute-force optimum: {res['most_likely'] == res['optimal']}")
    sched = pb.decode(res["most_likely"])
    print("  Recommended schedule:")
    for b, rs in sched.items():
        print(f"    {b:<18} -> {', '.join(rs)}")
    print(f"  Schedule cost: {res['most_likely_energy']:.2f}  (optimum {res['optimal_energy']:.2f})")
    no_risk = SchedulingProblem(batches, reactors, base_cost, conflicts, np.zeros(3))
    k_nr = int(np.argmin(no_risk.energies()))
    print(f"  Without quantum risk signal the optimiser would pick: {no_risk.decode(k_nr)}")
    report["schedule"] = {"assignment": sched, "p_optimal": res["p_optimal"],
                          "p_optimal_x_mixer_baseline": baseline["p_optimal"],
                          "p_feasible": res["p_feasible"], "matches_optimum": res["most_likely"] == res["optimal"],
                          "risk_blind_assignment": no_risk.decode(k_nr)}
    np.save("qubo_matrix.npy", pb.Q)  # hand-off to an annealer / gate-model backend

    # ---------- Module 3: VQE ----------
    banner("MODULE 3 | VQE molecular energy engine (process-chemistry primitive)")
    vqe = VQEMolecularEngine(layers=2).solve(rng)
    print(f"  H2 / STO-3G @ 0.735 Å, 2 qubits, hardware-efficient ansatz")
    print(f"  VQE total energy:   {vqe['vqe_total']:.6f} Ha")
    print(f"  Exact (FCI) energy: {vqe['exact_total']:.6f} Ha")
    print(f"  Error: {vqe['error_mHa']:.4f} mHa   chemical accuracy (<1.6 mHa): {vqe['chemical_accuracy']}")
    report["vqe"] = vqe

    banner("SUMMARY")
    print(f"  Closed loop complete in {time.time() - t0:.1f}s on a classical simulator.")
    print("  QUBO exported to qubo_matrix.npy; full report in quantum_manufacturing_report.json.")
    with open("quantum_manufacturing_report.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return report


if __name__ == "__main__":
    run()
