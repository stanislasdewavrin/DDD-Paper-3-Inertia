"""
Paper III — experiment 01: wavepacket propagation under 2-time Floquet
=======================================================================

We use the validated 2-step A/B Floquet rule
(paperI_foundations/code/99_demo_2step.py) and ask: do localised
wavepackets with momentum k0 actually translate at the predicted
group velocity v_g = dE/dk?

Setup:
  - 2D lattice, L x L periodic, two-component spinor.
  - Gaussian envelope in real space, multiplied by plane wave e^{i k0.x}.
  - Center the envelope at (L/2, L/2). Width sigma.
  - Apply N cycles of (A, B) and track:
      * centroid position <x>(t)
      * spread sigma_x(t)
      * total norm |psi|^2

We sweep k0_x for a few values and compare the measured drift speed
to the predicted v_g(k0) from the dispersion E(k) = |d(k)| with
  d(k) = (sin kx, sin ky, cos kx + cos ky + lambda cos(kx+ky))
The closest-approach Weyl-like points are at (0, pi) and (pi, 0)
where d = (0, 0, ±lambda); excitations near those points are massive
Dirac-like with rest mass m = lambda = 1/(2 pi).

What we measure here is just KINEMATIC: that the rule supports
propagating excitations with v_g = dE/dk. Time dilation is the
NEXT experiment (script 02).
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

PI = np.pi
LAMBDA = 1.0 / (2 * PI)


# ============================================================
# 2-step Floquet rule (lifted from 99_demo_2step.py)
# ============================================================
def step_A(psi, tau_A, L):
    """Matter step, H_A = sin(kx) sx + sin(ky) sy + (cos kx + cos ky) sz."""
    psi_a = psi[..., 0]; psi_b = psi[..., 1]
    a_k = np.fft.fft2(psi_a); b_k = np.fft.fft2(psi_b)
    kx = 2 * PI * np.fft.fftfreq(L); ky = 2 * PI * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dx = np.sin(KX); dy = np.sin(KY); dz = np.cos(KX) + np.cos(KY)
    d_norm = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-15)
    c = np.cos(d_norm * tau_A); s = np.sin(d_norm * tau_A)
    U00 = c - 1j * s * dz / d_norm
    U01 = -1j * s * (dx - 1j * dy) / d_norm
    U10 = -1j * s * (dx + 1j * dy) / d_norm
    U11 = c + 1j * s * dz / d_norm
    new_a_k = U00 * a_k + U01 * b_k
    new_b_k = U10 * a_k + U11 * b_k
    return np.stack([np.fft.ifft2(new_a_k), np.fft.ifft2(new_b_k)], axis=-1)


def step_B(psi, tau_B, L, lam=LAMBDA):
    """Gauge step, H_B = lambda * cos(kx+ky) sz."""
    psi_a = psi[..., 0]; psi_b = psi[..., 1]
    a_k = np.fft.fft2(psi_a); b_k = np.fft.fft2(psi_b)
    kx = 2 * PI * np.fft.fftfreq(L); ky = 2 * PI * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dz = lam * np.cos(KX + KY)
    phase = np.exp(-1j * dz * tau_B)
    return np.stack([np.fft.ifft2(phase * a_k),
                     np.fft.ifft2(np.conj(phase) * b_k)], axis=-1)


# ============================================================
# Wavepacket initialisation
# ============================================================
def make_wavepacket(L, k0, sigma, x0=None):
    """Build a Gaussian wavepacket centred at x0 with mean wavevector k0.
    The spinor structure is the lower-band eigenstate of d(k0)."""
    if x0 is None:
        x0 = (L / 2, L / 2)
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    env = np.exp(-((xs - x0[0])**2 + (ys - x0[1])**2) / (2 * sigma**2))
    plane = np.exp(1j * (k0[0] * xs + k0[1] * ys))
    # Use the spinor eigenstate of H_A + H_B at k = k0 (lower band).
    dx = np.sin(k0[0]); dy = np.sin(k0[1])
    dz = np.cos(k0[0]) + np.cos(k0[1]) + LAMBDA * np.cos(k0[0] + k0[1])
    d = np.array([dx, dy, dz])
    n = d / max(np.linalg.norm(d), 1e-15)
    # Lower-band eigenstate: |-> with sigma . n |-> = -|->
    # For n = (nx, ny, nz), |-> = (-(nx - i ny), 1 + nz)^T / norm  (when nz != -1)
    if abs(1 + n[2]) > 1e-9:
        u = np.array([-(n[0] - 1j * n[1]), 1 + n[2]], dtype=complex)
    else:  # n along -z, eigenstate is (1, 0)
        u = np.array([1.0, 0.0], dtype=complex)
    u /= np.linalg.norm(u)
    psi = np.zeros((L, L, 2), dtype=complex)
    psi[..., 0] = env * plane * u[0]
    psi[..., 1] = env * plane * u[1]
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi


def centroid(psi, L):
    """Return <x>, <y> using density |psi|^2 with periodic boundary handling."""
    rho = np.abs(psi[..., 0])**2 + np.abs(psi[..., 1])**2
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    # For periodic boundaries we use angle averaging
    angle_x = np.angle(np.sum(rho * np.exp(2j * PI * xs / L)))
    angle_y = np.angle(np.sum(rho * np.exp(2j * PI * ys / L)))
    cx = (angle_x / (2 * PI)) * L % L
    cy = (angle_y / (2 * PI)) * L % L
    return cx, cy


def spread(psi, L, cx, cy):
    """Gaussian-envelope spread sigma_x via the second moment of density."""
    rho = np.abs(psi[..., 0])**2 + np.abs(psi[..., 1])**2
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    dx = np.minimum(np.abs(xs - cx), L - np.abs(xs - cx))
    dy = np.minimum(np.abs(ys - cy), L - np.abs(ys - cy))
    var = np.sum(rho * (dx**2 + dy**2))
    return np.sqrt(var)


def predicted_vg(k0, lam=LAMBDA):
    """Group velocity v_g = grad_k E(k0), with E = |d(k)|."""
    kx, ky = k0
    dx = np.sin(kx); dy = np.sin(ky); dz = np.cos(kx) + np.cos(ky) + lam * np.cos(kx + ky)
    E = np.sqrt(dx**2 + dy**2 + dz**2)
    if E < 1e-12:
        return 0.0, 0.0
    # dE/dk_x = (1/E) (dx cos kx + dz (-sin kx - lam sin(kx+ky)))
    dE_dkx = (dx * np.cos(kx) + dz * (-np.sin(kx) - lam * np.sin(kx + ky))) / E
    dE_dky = (dy * np.cos(ky) + dz * (-np.sin(ky) - lam * np.sin(kx + ky))) / E
    return dE_dkx, dE_dky


# ============================================================
# Run the experiment
# ============================================================
def main():
    L = 64
    sigma = 5.0
    tau_A = 1.0; tau_B = 1.0
    n_cycles = 30  # shorter than L / (2 v) so packet stays localised

    # Sweep momenta near (kx, ky) = (pi, 0) gap point with small offset.
    # At k = (pi, 0): d = (0, 0, -lambda) so massive Dirac with m = lambda.
    # Move k slightly away: k = (pi - delta, 0).
    deltas = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80])
    # NOTE: k0 below is the centred *crystal* momentum, sampled around pi.

    results = []
    print(f"L={L} sigma={sigma} tau_A={tau_A} tau_B={tau_B} n_cycles={n_cycles}")
    print(f"Mass parameter: m = lambda = {LAMBDA:.6f}")
    print()
    print(f"{'delta':>8} {'k0x':>8} {'v_pred':>10} {'v_meas':>10} {'sigma_f':>10} "
          f"{'norm_drift':>12} {'E':>10}")
    for delta in deltas:
        k0 = (PI - delta, 0.0)  # crystal momentum
        psi = make_wavepacket(L, k0, sigma)
        norm0 = np.sqrt(np.sum(np.abs(psi)**2))
        cx0, cy0 = centroid(psi, L)
        sx0 = spread(psi, L, cx0, cy0)

        # Predicted velocity at k0
        vgx_pred, vgy_pred = predicted_vg(k0)
        # Theoretical energy
        dx = np.sin(k0[0]); dy = np.sin(k0[1])
        dz = np.cos(k0[0]) + np.cos(k0[1]) + LAMBDA * np.cos(k0[0] + k0[1])
        E = np.sqrt(dx**2 + dy**2 + dz**2)

        # Run N cycles
        track_x = []
        for it in range(n_cycles + 1):
            cx, cy = centroid(psi, L)
            track_x.append(cx)
            if it == n_cycles:
                break
            psi = step_A(psi, tau_A, L)
            psi = step_B(psi, tau_B, L)

        # Measured velocity from drift in x (per cycle = per unit time tau_A+tau_B)
        # Account for periodic wrap when measuring drift
        track_x = np.array(track_x)
        dx_drift = np.diff(track_x)
        dx_drift = np.where(dx_drift > L / 2, dx_drift - L, dx_drift)
        dx_drift = np.where(dx_drift < -L / 2, dx_drift + L, dx_drift)
        v_meas = float(np.mean(dx_drift) / (tau_A + tau_B))

        # Final spread + norm drift
        cxf, cyf = centroid(psi, L)
        sxf = spread(psi, L, cxf, cyf)
        normf = np.sqrt(np.sum(np.abs(psi)**2))
        norm_drift = float(normf / norm0 - 1)

        print(f"{delta:8.2f} {k0[0]:8.4f} {vgx_pred:10.4f} {v_meas:10.4f} "
              f"{sxf:10.4f} {norm_drift:12.4e} {E:10.4f}")

        results.append({
            "delta": float(delta),
            "k0x": float(k0[0]), "k0y": float(k0[1]),
            "v_pred": float(vgx_pred),
            "v_meas": v_meas,
            "sigma_initial": float(sx0),
            "sigma_final": float(sxf),
            "norm_drift": norm_drift,
            "E": float(E),
        })

    out = {
        "L": L, "sigma": sigma, "tau_A": tau_A, "tau_B": tau_B,
        "n_cycles": n_cycles, "lambda": LAMBDA,
        "results": results,
    }
    with open(DATA / "01_wavepacket_propagation.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved data to {DATA / '01_wavepacket_propagation.json'}")


if __name__ == "__main__":
    main()
