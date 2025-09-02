import numpy as np
import h5py

def gaussian_potential(x, y, A, x0, y0, sigma):
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))

def g(Ex, Ey, xg, yg, x, y):
    """Bilinear interpolation of Ex,Ey from grid to position (x,y)"""
    x = float(np.clip(x, xg[0], xg[-1]))
    y = float(np.clip(y, yg[0], yg[-1]))

    ix = int(np.clip(np.searchsorted(xg, x) - 1, 0, len(xg) - 2))
    iy = int(np.clip(np.searchsorted(yg, y) - 1, 0, len(yg) - 2))

    x1, x2 = xg[ix], xg[ix+1]
    y1, y2 = yg[iy], yg[iy+1]
    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)

    def bilerp(F):
        return ((1 - tx) * (1 - ty) * F[iy, ix] +
                 tx * (1 - ty) * F[iy, ix+1] +
                (1 - tx) *      ty  * F[iy+1, ix] +
                 tx *      ty  * F[iy+1, ix+1])

    return float(bilerp(Ex)), float(bilerp(Ey))

def boris_push(x, y, vx, vy, dt, q, m, Bz0, Ex, Ey, xg, yg):
    """One Boris step in 2D with uniform Bz."""
    Ex_p, Ey_p = g(Ex, Ey, xg, yg, x, y)

    vx_minus = vx + (q * Ex_p / m) * (dt / 2.0)
    vy_minus = vy + (q * Ey_p / m) * (dt / 2.0)

    t = (q * Bz0 / m) * (dt / 2.0)
    s = 2.0 * t / (1.0 + t*t)

    vpx = vx_minus - vy_minus * t
    vpy = vy_minus + vx_minus * t

    vx_plus = vx_minus - vpy * s
    vy_plus = vy_minus + vpx * s

    vx_new = vx_plus + (q * Ex_p / m) * (dt / 2.0)
    vy_new = vy_plus + (q * Ey_p / m) * (dt / 2.0)

    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    return x_new, y_new, vx_new, vy_new, Ex_p, Ey_p

nx, ny = 128, 128
Lx, Ly = 10.0, 10.0
Bz0 = 1.0
dt = 0.02
n_steps = 3000
q = m = 1.0

# Initial 
x0, y0 = -3.0, -1.5
vx0, vy0 = 0.8, 0.5

xg = np.linspace(-Lx/2, Lx/2, nx)
yg = np.linspace(-Ly/2, Ly/2, ny)
dx = xg[1] - xg[0]
dy = yg[1] - yg[0]
X, Y = np.meshgrid(xg, yg, indexing="xy")

phi = gaussian_potential(X, Y, 5.0, -2.0, 0.0, 1.2) \
    - gaussian_potential(X, Y, 5.0,  2.0, 0.0, 1.2)

Ex = -np.gradient(phi, dx, axis=1)
Ey = -np.gradient(phi, dy, axis=0)

t = np.arange(n_steps, dtype=np.float64) * dt
xs  = np.empty(n_steps, dtype=np.float64)
ys  = np.empty(n_steps, dtype=np.float64)
vxs = np.empty(n_steps, dtype=np.float64)
vys = np.empty(n_steps, dtype=np.float64)
Ex_p_hist = np.empty(n_steps, dtype=np.float64)
Ey_p_hist = np.empty(n_steps, dtype=np.float64)

x, y, vx, vy = float(x0), float(y0), float(vx0), float(vy0)
for i in range(n_steps):
    x, y, vx, vy, Ex_p, Ey_p = boris_push(x, y, vx, vy, dt, q, m, Bz0, Ex, Ey, xg, yg)
    xs[i], ys[i], vxs[i], vys[i] = x, y, vx, vy
    Ex_p_hist[i], Ey_p_hist[i] = Ex_p, Ey_p

out_file = "exp2.h5"  
with h5py.File(out_file, "w") as f:
    
    grp = f.create_group("grid")
    grp.create_dataset("x",   data=xg)
    grp.create_dataset("y",   data=yg)
    grp.create_dataset("phi", data=phi)
    grp.create_dataset("Ex",  data=Ex)
    grp.create_dataset("Ey",  data=Ey)
    grp.attrs["Bz"] = float(Bz0)

    grp["x"].attrs["units"]   = "m"
    grp["y"].attrs["units"]   = "m"
    grp["phi"].attrs["units"] = "V"
    grp["Ex"].attrs["units"]  = "V/m"
    grp["Ey"].attrs["units"]  = "V/m"
    grp.attrs["Bz_units"]     = "T"
    
    p = f.create_group("particle")
    p.create_dataset("t",  data=t);   p["t"].attrs["units"] = "s"
    p.create_dataset("x",  data=xs)
    p.create_dataset("y",  data=ys)
    p.create_dataset("vx", data=vxs)
    p.create_dataset("vy", data=vys)
    p.create_dataset("Ex_p", data=Ex_p_hist)
    p.create_dataset("Ey_p", data=Ey_p_hist)
    p.attrs.update({"q": q, "m": m, "dt": dt, "n_steps": n_steps, "integrator": "Boris"})

