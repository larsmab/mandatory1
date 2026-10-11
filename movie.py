import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Wave2D import Wave2D_Neumann

sol = Wave2D_Neumann()
data = sol(50, 100, cfl=1/np.sqrt(2), mx=2, my=2, store_data=4)
xij, yij = sol.xij, sol.yij
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 4))
frames = []
for n, val in data.items():
    frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2)
    frames.append([frame])
ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
ani.save("report/neumannwave.gif", writer="pillow", fps=15)