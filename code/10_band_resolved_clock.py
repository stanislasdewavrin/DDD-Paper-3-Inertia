"""
Paper III - exp 10: BAND-RESOLVED operational comoving clock rate.

Fix to the high-gamma upturn pathology of fig3:
the previous projection picked the LARGER spinor component (A_a or A_b),
which mixes Floquet bands. Here we project onto the lower-band eigenstate
at k0 explicitly:
    A_band(t) = u_-(k0).conj() . [proj_k0(psi(t))]
This isolates the lower-band contribution exactly (modulo wavepacket
spread), and gives a clean exp(-i eps_-(k0) t) evolution in lab frame.

The Galilean shift exp(-i delta v_g t) then yields omega_comov = m/gamma
at all gamma where the wavepacket remains predominantly lower-band.
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi
LAMBDA = 1.0 / (2 * PI)


def step_A(psi, tau_A, L):
    a_k = np.fft.fft2(psi[..., 0]); b_k = np.fft.fft2(psi[..., 1])
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
    a_k = np.fft.fft2(psi[..., 0]); b_k = np.fft.fft2(psi[..., 1])
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
    UB = np.array([[np.exp(-1j*dzB*tau_B), 0],
                   [0, np.exp(1j*dzB*tau_B)]], dtype=complex)
    UF = UB @ UA
    evals, evecs = np.linalg.eig(UF)
    T = tau_A + tau_B
    quasi = -np.angle(evals) / T
    if quasi[0] < quasi[1]:
        return float(quasi[0]), float(quasi[1]), evecs[:, 0], evecs[:, 1]
    return float(quasi[1]), float(quasi[0]), evecs[:, 1], evecs[:, 0]


def floquet_vg(kx0, ky0, h=1e-4):
    lo_p, _, _, _ = floquet_diag(kx0+h, ky0)
    lo_m, _, _, _ = floquet_diag(kx0-h, ky0)
    return (lo_p - lo_m) / (2*h)


def fix_phase(u):
    """Phase convention: larger-modulus component real and non-negative."""
    if abs(u[0]) >= abs(u[1]):
        return u * np.exp(-1j * np.angle(u[0]))
    return u * np.exp(-1j * np.angle(u[1]))


def make_wp(L, k0, sigma):
    eps_lo, _, u_lo, _ = floquet_diag(k0[0], k0[1])
    u_lo = fix_phase(u_lo)
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    env = np.exp(-((xs-L/2)**2 + (ys-L/2)**2) / (2*sigma**2))
    plane = np.exp(1j*(k0[0]*xs + k0[1]*ys))
    psi = np.zeros((L, L, 2), dtype=complex)
    psi[..., 0] = env * plane * u_lo[0]
    psi[..., 1] = env * plane * u_lo[1]
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi, eps_lo, u_lo


def project_k0_band(psi, k0, L, u_lo):
    """Band-resolved projection: <u_-, k0 | psi> = u_-^* . sum_x e^{-i k0.x} psi(x)."""
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    phase = np.exp(-1j*(k0[0]*xs + k0[1]*ys))
    A_a = np.sum(phase * psi[..., 0])
    A_b = np.sum(phase * psi[..., 1])
    # inner product with lower-band eigenstate (conjugate)
    return complex(np.conj(u_lo[0]) * A_a + np.conj(u_lo[1]) * A_b)


def run_one(L, sigma, n_cycles, deltas, tau_A=1.0, tau_B=1.0):
    T_per_cycle = tau_A + tau_B
    eps_lo_gap, _, _, _ = floquet_diag(PI, 0.0)
    m_floq = abs(eps_lo_gap)
    results = []
    for delta in deltas:
        k0 = (PI - delta, 0.0)
        eps_lo, _, _, _ = floquet_diag(k0[0], k0[1])
        E_floq = abs(eps_lo)
        gamma_floq = E_floq / m_floq if m_floq > 0 else 0.0
        v_g = floquet_vg(k0[0], 0.0)
        m_over_gamma = m_floq / gamma_floq if gamma_floq > 0 else m_floq

        psi, _, u_lo = make_wp(L, k0, sigma)
        proj_lab = []
        for it in range(n_cycles + 1):
            A_band = project_k0_band(psi, k0, L, u_lo)
            proj_lab.append(A_band)
            if it == n_cycles:
                break
            psi = step_A(psi, tau_A, L)
            psi = step_B(psi, tau_B, L)

        ts = np.arange(n_cycles + 1) * T_per_cycle
        proj_co = np.array(proj_lab) * np.exp(-1j * delta * v_g * ts)
        ph = np.unwrap(np.angle(proj_co))
        i0, i1 = 5, n_cycles - 4
        slope = np.polyfit(ts[i0:i1+1], ph[i0:i1+1], 1)[0]
        omega_co_meas = abs(float(slope))
        rel_err = (omega_co_meas - m_over_gamma) / m_over_gamma if m_over_gamma > 0 else 0.0

        # also compute amplitude purity: how much of the projection magnitude
        # remained in the lower band over time
        amp = np.abs(np.array(proj_lab))
        purity = float(amp[-1] / amp[0]) if amp[0] > 0 else 0.0

        results.append({
            "delta": float(delta),
            "E_floq": float(E_floq),
            "gamma_floq": float(gamma_floq),
            "v_g": float(v_g),
            "m_over_gamma_pred": float(m_over_gamma),
            "omega_comoving_meas": omega_co_meas,
            "rel_err": float(rel_err),
            "lower_band_purity": purity,
        })
        print(f"  L={L} d={delta:5.2f} g={gamma_floq:6.3f} pred={m_over_gamma:.5f} meas={omega_co_meas:.5f} err={rel_err:+7.3%} purity={purity:.3f}", flush=True)
    return {"L": L, "sigma": sigma, "n_cycles": n_cycles,
            "tau_A": tau_A, "tau_B": tau_B, "m_floq": m_floq,
            "lambda": LAMBDA, "method": "band-resolved projection",
            "results": results}


def main():
    deltas = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                       0.60, 0.80, 1.00, 1.20, 1.40])
    n_cycles = 60
    runs = []
    for L, sigma in [(128, 12.0), (256, 24.0)]:
        print(f"\n=== L={L} sigma={sigma} ncycles={n_cycles} (band-resolved) ===", flush=True)
        runs.append(run_one(L, sigma, n_cycles, deltas))
    out = {"description": "Band-resolved operational clock rate (fix for high-gamma upturn)",
           "deltas": deltas.tolist(), "runs": runs}
    with open(DATA / "10_band_resolved_clock.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {DATA / '10_band_resolved_clock.json'}")


if __name__ == "__main__":
    main()
