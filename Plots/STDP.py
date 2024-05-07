import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from Plot_Helper import *

fig = plt.figure()

fig.set_figwidth(14)
fig.set_figheight(4)

a1 = fig.add_subplot(1, 3, 1)
a2 = fig.add_subplot(1, 3, 2)


a4 = fig.add_subplot(1, 3, 3, projection='3d')

t = np.arange(-2, 2, 0.001)

x = np.arange(0, 0.6, 0.01)

a1.plot(x, ltp(x), c=colorLTP, label='LTP')
a1.plot(x, -ltd(x), c=colorLTD, label='LTD')
a1.plot(x, ltp(x)-ltd(x), c=colorSTDP, label='LTP+LTD')

a1.axhline(0, c='black', linewidth=1)
a1.spines[['right', 'top', 'bottom']].set_visible(False)
a1.set_xticks([])
a1.set_yticks([])

a1.set_ylabel("$\Delta$ w")
a1.set_xlim([0, 0.6])

a1.text(0.3, 0.4, 'LTP', c=colorLTP)
a1.text(0.3, -0.25, 'LTD', c=colorLTD)
a1.text(0.3, 0.15, 'LTP+LTD', c=colorSTDP)

a1.text(0.62, 0.0, '$Ca^{2+}$', c=(0, 0, 0, 1))

a2.plot(t, ca(t, 0), c=colorCA, label=' ')
a2.plot(t, stdp(ca(t, 0)), c=colorSTDP, label='low GABA')

a2.plot(t, ca(t, 0.4), c=colorCA, label=' ', alpha=0.6)
a2.plot(t, stdp(ca(t, 0.4)), c=colorSTDP, label='med. GABA', alpha=0.6)

a2.plot(t, ca(t, 0.665), c=colorCA, label=' ', alpha=0.2)
a2.plot(t, stdp(ca(t, 0.665)), c=colorSTDP, label='high GABA', alpha=0.2)

a2.axhline(0, c='black', linewidth=1)
a2.axvline(0, c='black', linewidth=1)
a2.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
a2.set_xticks([])
a2.set_yticks([])

a2.set_xlim([-2, 2])

a2.text(0.8, 0.5, '$Ca^{2+}$', c=colorCA)
a2.text(0.8, 0.15, '$\Delta$ w', c=colorSTDP)

a2.text(2.2, 0.0, '$\Delta$ t', c=(0, 0, 0, 1))

a2.legend()




X = np.arange(-2.2, 2.2, 0.001)
G = np.arange(0,0.9,0.001)
X, G = np.meshgrid(X, G)
Z = stdp(ca(X, G))



grad = LinearSegmentedColormap.from_list('grad', (
    (0.000, colorLTD),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, colorLTP)))



surf = a4.plot_surface(X, G, Z, cmap=grad, linewidth=0, antialiased=False,  vmin=-0.1, vmax=0.1)

data = stdp(ca(t, 0))
a4.plot(t, t*0, data, c=colorSTDP, linewidth=5, zorder=1.0)

a4.set_xlabel('$\Delta$ t')
a4.set_ylabel('GABA')
a4.set_zlabel('$\Delta$ w')

a4.set_xticks([])
a4.set_yticks([])
a4.set_zticks([])



a1.text(x=0.0-0.05, y=0.6, s='A', size=20, weight='bold')
a1.text(x=0.7-0.05, y=0.6, s='B', size=20, weight='bold')
a1.text(x=1.4-0.05, y=0.6, s='C', size=20, weight='bold')

a1.text(x=3, y=0.6, s=' ', size=20, weight='bold')

plt.tight_layout()
plt.show()


