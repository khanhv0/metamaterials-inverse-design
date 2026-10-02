import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("simulation_results.csv")[["poisson", "young", "volume_frac"]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["poisson"], df["young"], df["volume_frac"])
ax.set_xlabel('Effective Poisson')
ax.set_ylabel('Effective Young')
ax.set_zlabel('Volume fraction')

plt.show()