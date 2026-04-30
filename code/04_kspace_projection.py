"""
Paper III — experiment 04: k-space projection for clean phase tracking
=======================================================================

Project the wavefunction onto the central k0 plane wave at each tick:
    A_lab(t) = sum_x exp(-i k0 . x) psi_a(x, t)
This isolates the k0 mode, which evolves as exp(-i eps_lo(k0) t)
exactly. Its phase rate gives the LAB-FRAME energy E = |eps_lo(k0)|.

For the COMOVING-FRAME (proper-time) rate, we need to subtract the
k0 . v_g . t contribution. Equivalently, the smooth-envelope phase
evolution at the moving frame:
    omega_comoving = E - p . v_g   where p = k0 - pi (offset from gap)

For Lorentz dispersion E^2 = m^2 + (p/2)^2 with v_g = p/(4E):
    omega_comoving = m^2 / E = m / gamma

This script:
  1. measures lab-frame E at k0 (no centroid tracking) -> matches eps_lo perfectly
  2. computes comoving rate analytically from E_floq, p, v_g (Floquet)
  3. verifies the Lorentz form E^2 = m^2 + (p/2)^2 numerically
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi
LAMBDA = 1.0 / (2 * PI)


def step_A(psi, tau_A, L):
    psi_a = psi[..., 0]; psi_b = psi[..., 1]
    a_k = np.fft.fft2(psi_a); b_k = np.fft.fft2(psi_b)
    kx = 2*PI*np.fft.fftfreq(L); ky = 2*PI*np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dx = np.sin(KX); dy = np.sin(KY); dz = np.cos(KX)+np.cos(KY)
    dn = np.sqrt(dx**2+dy**2+dz**2+1e-15)
    c = np.cos(dn*tau_A); s = np.sin(dn*tau_A)
    U00 = c - 1j*s*dz/dn
    U01 = -1j*s*(dx-1j*dy)/dn
    U10 = -1j*s*(dx+1j*dy)/dn
    U11 = c + 1j*s*dz/dn
    return np.stack([np.fft.ifft2(U00*a_k+U01*b_k),
                     np.fft.ifft2(U10*a_k+U11*b_k)], axis=-1)


def step_B(psi, tau_B, L, lam=LAMBDA):
    psi_a = psi[..., 0]; psi_b = psi[..., 1]
    a_k = np.fft.fft2(psi_a); b_k = np.fft.fft2(psi_b)
    kx = 2*PI*np.fft.fftfreq(L); ky = 2*PI*np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dz = lam*np.cos(KX+KY)
    ph = np.exp(-1j*dz*tau_B)
    return np.stack([np.fft.ifft2(ph*a_k),
                     np.fft.ifft2(np.conj(ph)*b_k)], axis=-1)


def floquet_diag(kx, ky, tau_A=1.0, tau_B=1.0, lam=LAMBDA):
    dxA = np.sin(kx); dyA = np.sin(ky); dzA = np.cos(kx)+np.cos(ky)
    nA = np.sqrt(dxA**2+dyA**2+dzA**2+1e-15)
    cA = np.cos(nA*tau_A); sA = np.sin(nA*tau_A)
    UA = np.array([[cA-1j*sA*dzA/nA, -1j*sA*(dxA-1j*dyA)/nA],
                   [-1j*sA*(dxA+1j*dyA)/nA, cA+1j*sA*dzA/nA]], dtype=complex)
    dzB = lam*np.cos(kx+ky)
    UB = np.array([[np.exp(-1j*dzB*tau_B), 0],[0, np.exp(1j*dzB*tau_B)]], dtype=complex)
    UF = UB @ UA
    evals, evecs = np.linalg.eig(UF)
    T = tau_A + tau_B
    quasi = -np.angle(evals)/T
    if quasi[0] < quasi[1]:
        return float(quasi[0]), float(quasi[1]), evecs[:, 0]
    return float(quasi[1]), float(quasi[0]), evecs[:, 1]


def make_wavepacket_floquet(L, k0, sigma):
    eps_lo, _, u_lo = floquet_diag(k0[0], k0[1])
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    x0 = (L/2, L/2)
    env = np.exp(-((xs-x0[0])**2 + (ys-x0[1])**2) / (2*sigma**2))
    plane = np.exp(1j*(k0[0]*xs + k0[1]*ys))
    psi = np.zeros((L, L, 2), dtype=complex)
    psi[..., 0] = env * plane * u_lo[0]
    psi[..., 1] = env * plane * u_lo[1]
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi, eps_lo


def project_k0(psi, k0, L):
    """Compute < k0, lower-band | psi >.
    Specifically: spatial Fourier amplitude of psi_a at exact k = k0.
    For an eigenstate at k0, this gives a complex phasor that rotates
    at exp(-i eps_lo t)."""
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    phase = np.exp(-1j*(k0[0]*xs + k0[1]*ys))
    A_a = np.sum(phase * psi[..., 0])
    A_b = np.sum(phase * psi[..., 1])
    return complex(A_a), complex(A_b)


def main():
    L = 64
    sigma = 6.0  # tighter k -> better phase tracking
    tau_A = 1.0; tau_B = 1.0
    T_per_cycle = tau_A + tau_B
    n_cycles = 80

    eps_lo_gap, _, _ = floquet_diag(PI, 0.0)
    m_floq = abs(eps_lo_gap)
    print(f"Floquet rest mass m = {m_floq:.6f}")
    print(f"Lorentz prediction: E^2 = m^2 + (p/2)^2, c_eff = 1/2")
    print()

    deltas = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

    print(f"{'delta':>6} {'E_floq':>10} {'E_Lor':>10} {'gamma_floq':>10} "
          f"{'omega_lab_meas':>16} {'omega_lab_pred':>16}  {'m^2/E':>10}")
    print("(omega_lab_meas: phase rate of <k0|psi(t)> from simulation)")
    print("(omega_lab_pred: -eps_lo from Floquet diagonalization)")
    print("(m^2/E: comoving-frame proper-time rate, for reference)")
    print()

    results = []
    for delta in deltas:
        k0 = (PI - delta, 0.0)
        eps_lo, _, _ = floquet_diag(k0[0], k0[1])
        E_floq = abs(eps_lo)
        E_Lor = np.sqrt(m_floq**2 + (delta/2)**2)
        gamma_floq = E_floq / m_floq
        m2_over_E = m_floq**2 / E_floq

        psi, _ = make_wavepacket_floquet(L, k0, sigma)
        # Track <k0|psi(t)> phase
        proj_phase = []
        for it in range(n_cycles + 1):
            A_a, A_b = project_k0(psi, k0, L)
            # Use psi_a component (assumes u_lo[0] != 0; for k0=(pi,0), u_lo=(1,0))
            # For other k0, A_a may be small - we use whichever has larger amplitude
            if abs(A_a) > abs(A_b):
                proj_phase.append(np.angle(A_a))
            else:
                proj_phase.append(np.angle(A_b))
            if it == n_cycles:
                break
            psi = step_A(psi, tau_A, L)
            psi = step_B(psi, tau_B, L)

        proj_phase = np.unwrap(np.array(proj_phase))
        ts = np.arange(n_cycles + 1) * T_per_cycle
        i0, i1 = 5, n_cycles - 4
        slope = np.polyfit(ts[i0:i1+1], proj_phase[i0:i1+1], 1)[0]
        omega_lab_meas = abs(float(slope))
        omega_lab_pred = E_floq

        print(f"{delta:6.3f} {E_floq:10.6f} {E_Lor:10.6f} {gamma_floq:10.4f} "
              f"{omega_lab_meas:16.6f} {omega_lab_pred:16.6f}  {m2_over_E:10.6f}")

        results.append({
            "delta": float(delta),
            "k0x": float(k0[0]),
            "E_floq": float(E_floq),
            "E_Lorentz_pred": float(E_Lor),
            "gamma_floq": float(gamma_floq),
            "omega_lab_meas": omega_lab_meas,
            "omega_lab_pred": omega_lab_pred,
            "m2_over_E": float(m2_over_E),
        })

    out = {"L": L, "sigma": sigma, "n_cycles": n_cycles,
           "tau_A": tau_A, "tau_B": tau_B, "m_floq": m_floq,
           "lambda": LAMBDA, "results": results}
    with open(DATA / "04_kspace_projection.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved to {DATA / '04_kspace_projection.json'}")


if __name__ == "__main__":
    main()
