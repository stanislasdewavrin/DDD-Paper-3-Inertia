"""Paper III — group velocity control:
compare v_g from dispersion derivative against centroid drift in simulation."""
import json, numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi
LAMBDA = 1.0/(2*PI)


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
    quasi = -np.angle(evals)/(tau_A+tau_B)
    if quasi[0] < quasi[1]:
        return float(quasi[0]), float(quasi[1]), evecs[:, 0]
    return float(quasi[1]), float(quasi[0]), evecs[:, 1]


def vg_disp(kx, ky, h=1e-4):
    lo_p, _, _ = floquet_diag(kx+h, ky)
    lo_m, _, _ = floquet_diag(kx-h, ky)
    return (lo_p - lo_m) / (2*h)


def make_wp(L, k0, sigma):
    eps_lo, _, u_lo = floquet_diag(k0[0], k0[1])
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    env = np.exp(-((xs-L/2)**2 + (ys-L/2)**2) / (2*sigma**2))
    plane = np.exp(1j*(k0[0]*xs + k0[1]*ys))
    psi = np.zeros((L, L, 2), dtype=complex)
    psi[..., 0] = env * plane * u_lo[0]
    psi[..., 1] = env * plane * u_lo[1]
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi


def centroid(psi, L):
    rho = np.abs(psi[..., 0])**2 + np.abs(psi[..., 1])**2
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    ax = np.angle(np.sum(rho * np.exp(2j*PI*xs/L)))
    return float((ax/(2*PI)) * L % L)


def main():
    L = 128; sigma = 12.0; n_cycles = 40; T = 2.0
    deltas = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

    print(f"{'delta':>6} {'v_g_disp':>10} {'v_g_drift':>10} {'rel_err':>8}")
    rows = []
    for delta in deltas:
        k0 = (PI - delta, 0.0)
        v_disp = vg_disp(k0[0], 0.0)
        psi = make_wp(L, k0, sigma)
        cx = [centroid(psi, L)]
        for it in range(n_cycles):
            psi = step_A(psi, 1.0, L); psi = step_B(psi, 1.0, L)
            cx.append(centroid(psi, L))
        cx = np.array(cx)
        # unwrap periodic jumps
        d = np.diff(cx)
        d = np.where(d > L/2, d - L, d)
        d = np.where(d < -L/2, d + L, d)
        ts = np.arange(n_cycles+1) * T
        # Drift v from average of differences (avoiding noise from initial transient)
        v_drift = float(np.mean(d[5:]) / T)
        rel = (v_drift - v_disp) / v_disp if v_disp != 0 else 0.0
        rows.append({"delta": float(delta), "v_g_dispersion": float(v_disp),
                     "v_g_centroid_drift": v_drift, "rel_err": float(rel)})
        print(f"{delta:6.2f} {v_disp:10.4f} {v_drift:10.4f} {rel:8.2%}")

    out = {"L": L, "sigma": sigma, "n_cycles": n_cycles, "T": T, "results": rows}
    with open(DATA / "07_vg_control.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
