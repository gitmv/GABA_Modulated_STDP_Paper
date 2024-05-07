import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.image as image
from Plot_Helper import *

fig, ax = plt.subplots(1, 2)

fig.set_figwidth(10)
fig.set_figheight(4.5)

x = np.arange(-1, 1, 0.001)

for i, a in enumerate(ax):
    a.spines[['right', 'top']].set_visible(False)  # , 'bottom'
    #a.spines['left'].set_color('lightgray')
    #a.spines['bottom'].set_color('lightgray')

      # 1.5

    a.set_ylim([-0.01, 2.5])

ax[0].set_ylabel("spike probability")
ax[0].set_xlabel("voltage")
ax[1].set_xlabel("voltage")
ax[0].set_yticks([])
ax[1].set_yticks([1])

ax[0].plot(x, F_E(x), c=colorE, linewidth=2)
ax[0].set_xticks([0, 0.8])
ax[0].set_xlim([-0.8/3, 0.8])


ax[1].plot(x, F_I(x), c=colorI, linewidth=2)
ax[1].set_xticks([0, 0.3])
ax[1].set_xlim([-0.1, 0.3])

ax[0].text(-0.0, 0.2, 'curvature important\nfor stability!', c=color3, rotation=45, ha='center')

ax[0].plot(x, np.clip(np.power(np.clip(x, 0, None)*2, 0.60416), 0, 1), linestyle='dashed', c=(0, 0, 0, 0.3)) #=(0, 0.8, 0.8, 0.7)
ax[1].plot(x, np.clip(np.tanh(x * 15.5), 0, 1), linestyle='dashed', c=(0, 0, 0, 0.3))

ax[0].scatter([0.68], [2.0], c=colorE, marker='^', s=3000)
ax[1].scatter([0.25], [2.0], c=colorI, marker='o', s=3000)


img = image.imread('../Data/PC_WT_FI.png')
axicon = fig.add_axes([0.09, 0.48, 0.4, 0.4])
axicon.imshow(img, interpolation='bilinear')
axicon.set_xticks([])
axicon.set_yticks([])
axicon.spines[['right', 'top', 'left', 'bottom']].set_visible(False)


img = image.imread('../Data/IN_WT_FI.png')
axicon = fig.add_axes([0.50, 0.48, 0.4, 0.4])
axicon.imshow(img, interpolation='bilinear')
axicon.set_xticks([])
axicon.set_yticks([])
axicon.spines[['right', 'top', 'left', 'bottom']].set_visible(False)



ax[0].text(x=-0.35, y=2.3, s='A', size=20, weight='bold')
ax[0].text(x=-0.35+1.28, y=2.3, s='B', size=20, weight='bold')

ax[0].text(x=10, y=0, s=' ', size=20, weight='bold')



plt.tight_layout()
plt.show()

#neurons.spike = np.power(np.clip(neurons.voltage * self.mul, 0.0, None), self.exp) > neurons.vector("uniform")

#neurons.spike = (self.avg_act * self.avg_inh / self.target_activity) > neurons.vector('uniform')