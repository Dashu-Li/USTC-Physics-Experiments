import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpmath import mp, mpf, mpc, quad, besselj, fabs

mp.dps = 25  # 精度设置

# 正七边形的前顶点（单位圆上均匀分布）
n = 7
a_k = [mp.exp(2j * mp.pi * k / n) for k in range(n)]

# SC 映射函数（将单位圆内点映射到正七边形）
def sc_integral(zeta):
    def integrand(xi):
        product = mpf(1.0)
        for ak in a_k:
            product *= (1 - xi / ak) ** (-5 / 7)
        return product / xi
    return quad(integrand, [0, zeta])

# 生成单位圆内网格（极坐标）
theta = np.linspace(0, 2*np.pi, 150)
r = np.linspace(0, 0.98, 120)
R, T = np.meshgrid(r, theta)
ZETA = R * np.exp(1j * T)
Z = np.zeros_like(ZETA, dtype=np.complex128)
U = np.zeros_like(ZETA, dtype=np.float64)

# 模态选择（主模态）
from scipy.special import jn_zeros
m, s = 0, 1
alpha_ms = float(jn_zeros(m, s)[-1])

# 对每个点计算 z 与 u(z)
for i in range(ZETA.shape[0]):
    for j in range(ZETA.shape[1]):
        zeta = mpc(ZETA[i, j].real, ZETA[i, j].imag)
        z = sc_integral(zeta)
        z_mod = fabs(zeta)
        phi_val = besselj(m, alpha_ms * z_mod)
        Z[i, j] = complex(z.real, z.imag)
        U[i, j] = float(phi_val)

# 绘图：正七边形中的模态
X = Z.real
Y = Z.imag
fig, ax = plt.subplots(figsize=(6.5, 6.5))
c = ax.tricontourf(X.flatten(), Y.flatten(), U.flatten(), levels=100, cmap=cm.viridis)
fig.colorbar(c, ax=ax, label='Re[$u(z)$]')
ax.set_title('正七边形波导主模态 $u(z)$', fontsize=14)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.show()
