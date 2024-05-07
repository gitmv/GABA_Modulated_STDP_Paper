import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Plot_Helper import *

p = 1.83
g = 0.86
h = 0.05

fig = plt.figure()

fig.set_figwidth(14)
fig.set_figheight(4)

a1 = fig.add_subplot(1, 3, 1)
a2 = fig.add_subplot(1, 3, 2)
a3 = fig.add_subplot(1, 3, 3)


dt = 0.1
step = 0.0001

for x in np.arange(theta-dt, theta, step):
    a1.fill_between([x, x+step], [M(x), M(x+step)], facecolor=colorLTP, alpha=np.power(M(x), 2))

for x in np.arange(theta, theta+dt, step):
    a1.fill_between([x, x+step], [M(x), M(x+step)], facecolor=colorLTD, alpha=np.power(M(x)/min_m, 2))


x = np.arange(theta-dt, theta+dt, 0.0001)
a1.plot(x, M(x), c='black', linewidth=1)#color3


a1.axhline(0, c='black', linewidth=0.8)
a1.spines[['right', 'top', 'bottom']].set_visible(False)
a1.set_xticks([])
#a1.set_yticks([0.4, 1])#-0.2,
a1.set_yticks([-0.15, 1], [r'$\mu$','1'])#-0.2,

a1.set_ylabel("STDP Modulation M(t)")
a1.set_xlim([theta-dt, theta+dt])
a1.set_ylim([-1.5, 1.5])

a1.text(theta+dt+0.005, 0.0, 'GABA', c=(0, 0, 0, 1))

g = 0.07

a1.plot([theta-g, theta-g], [0-h, -0.5], c=(0.8, 0.8, 0.8, 1), linewidth=1)
a1.plot([theta-g/2, theta-g/2], [0-h, -0.5], c=(0.8, 0.8, 0.8, 1), linewidth=1)
a1.plot([theta, theta], [M(theta)-h, -0.5], c=(0.8, 0.8, 0.8, 1), linewidth=1)
a1.plot([theta+g/2, theta+g/2], [M(theta+g/2)-h, -0.5], c=(0.8, 0.8, 0.8, 1), linewidth=1)
a1.plot([theta+g, theta+g], [M(theta+g)-h, -0.5], c=(0.8, 0.8, 0.8, 1), linewidth=1)

a1.scatter([theta-g], [-0.9], s=1000000*np.power(theta-g, 8), color=colorE)
a1.scatter([theta-g/2], [-0.9], s=1000000*np.power(theta-g/2, 8), color=colorE)
a1.scatter([theta], [-0.9], s=1000000*np.power(theta, 8), color=colorE)
a1.scatter([theta+g/2], [-0.9], s=1000000*np.power(theta+g/2, 8), color=colorE)
a1.scatter([theta+g], [-0.9], s=1000000*np.power(theta+g, 8), color=colorE)

a1.arrow(theta-g, -1.2, +0.05, 0, head_width=0.05, head_length=0.005, color=(0, 0, 0, 1))
a1.arrow(theta+g, -1.2, -0.05, 0, head_width=0.05, head_length=0.005, color=(0, 0, 0, 1))

a1.text(theta, -1.4, 'active neuron cluster size regulation', c=(0, 0, 0, 1), ha='center')

a1.text(theta, 0.2, '$\Theta$', c=(0, 0, 0, 1), size=20, ha='center')

a1.plot([theta-g*1.3, theta+g*1.3, theta+g*1.3, theta-g*1.3, theta-g*1.3], [-0.6, -0.6, -1.49, -1.49, -0.6], c='black', linewidth=1)

a1.plot([0.26, 0.26], [-0.06, +0.06], c='black', linewidth=1)
a1.text(0.26, -0.2, '0.25', ha='center', va='top', size=8)

a1.plot([0.42, 0.42], [-0.06, +0.06], c='black', linewidth=1)
a1.text(0.42, 0.2, '0.43', ha='center', size=8)


a2.set_xticks([])
a2.set_yticks([])
a2.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

o = 1.3
a2.plot([0, 0], [4-o, 2.5-o], c='black', linewidth=0.8)
a2.plot([-1, 1], [3-o, 3-o], c='black', linewidth=0.8)
a2.text(1.1, 3-o, '$\Delta$ t', c=(0, 0, 0, 1))#, ha='center'
a2.text(0, 2.75, '$\Delta$ w', c=(0, 0, 0, 1), ha='center')
a2.text(-0.2, 3-o+0.2, 'M(t)=1', c=(0, 0, 0, 1), ha='right')

a2.plot([0.5, 0.5], [3-o-0.03, 3-o+0.02], c='black', linewidth=1)
a2.text(0.5, 3-o-0.3, '1', ha='center', size=8)

a2.plot([0, 0], [-.5, 1], c='black', linewidth=0.8)
a2.plot([-1, 1], [0, 0], c='black', linewidth=0.8)
a2.text(1.1, 0.0, '$\Delta$ t', c=(0, 0, 0, 1))
a2.text(-0.2, 0+0.2, 'M(t)=0', c=(0, 0, 0, 1), ha='right')

a2.plot([0.5, 0.5], [0-0.03, 0+0.02], c='black', linewidth=1)
#a2.text(0.5, 0-0.3, '1', ha='center', size=8)

a2.plot([0, 0], [-3.5+o, -2+o], c='black', linewidth=0.8)
a2.plot([-1, 1], [-3+o, -3+o], c='black', linewidth=0.8)
a2.text(1.1, -3+o, '$\Delta$ t', c=(0, 0, 0, 1))
a2.text(-0.2, -3+o+0.2, 'M(t)=-0.15', c=(0, 0, 0, 1), ha='right')

a2.plot([0.5, 0.5], [-3+o-0.03, -3+o+0.02], c='black', linewidth=1)
#a2.text(0.5, -3+o-0.3, '1', ha='center', size=8)

a2.set_xlim([-2.7, 2.7])
a2.set_ylim([-2.2, 2.7])


rect1 = matplotlib.patches.Rectangle([0.25, 3-o], 0.5, 0.8, color=colorSTDP)
rect2 = matplotlib.patches.Rectangle([0.25, 0], 0.5, 0.01, color=colorSTDP)
rect3 = matplotlib.patches.Rectangle([0.25, -3+o], 0.5, -0.2, color=colorSTDP)

a2.add_patch(rect1)
a2.add_patch(rect2)
a2.add_patch(rect3)









a3.spines[['right', 'top']].set_visible(False)


a3.set_yticks([0, target_act], ['0', 'h'])

a3.set_xlabel("t")
a3.set_ylabel("excitatory spike rate ( $\widehat{=}$ GABA)")

for y in np.arange(theta, theta*2, 0.0001):
    a3.axhline(y/theta*target_act, c=colorLTD, alpha=np.power(M(y)/min_m, 2))

for y in np.arange(0, theta, 0.0001):
    a3.axhline(y/theta*target_act, c=colorLTP, alpha=np.power(M(y), 2))


a3.set_ylim([0, target_act*2])

data = np.load('../Data/Char/E_np.mean(spike).npy')

a3.plot(np.arange(0, 200, 1), data[10000:10200], linewidth=0.5, c=colorE)
a3.plot(np.arange(250, 450, 1), data[30000:30200], linewidth=0.5, c=colorE)
a3.plot(np.arange(500, 700, 1), data[50000:50200], linewidth=0.5, c=colorE)

#a3.set_xticks([0, 250, 500], ['10k', '30k', '50k'])
a3.text(100, 0+0.001, '10k\n200 steps', c=(0, 0, 0, 1), ha='center', size=8)
a3.text(350, 0+0.001, '30k\n200 steps', c=(0, 0, 0, 1), ha='center', size=8)
a3.text(600, 0+0.001, '50k\n200 steps', c=(0, 0, 0, 1), ha='center', size=8)

a3.set_xticks([])

#a3.axhline(0-0.01, xmin=0, xmax=200, color='black', linewidth=2)# , linestyle='--'

#a3.arrow(100, 0, 0, +target_act, head_width=0.1, head_length=0.1, color=(0, 0, 0, 1))
#a3.arrow(400, 0, 0, +target_act, head_width=0.1, head_length=0.1, color=(0, 0, 0, 1))

#a3.plot(np.random.rand(500)*theta*2/5+theta-0.06, linewidth=0.5)
#a3.set_ylim([0+0.2, theta*2-0.2])

#a3.axhline(theta)


a1.text(x=theta-dt-0.01-0.03, y=1.6, s='A', size=20, weight='bold')
a1.text(x=theta-dt-0.01+0.3, y=1.6, s='B', size=20, weight='bold')
a1.text(x=theta-dt-0.01+0.46, y=1.6, s='C', size=20, weight='bold')

a1.text(x=3, y=10, s=' ', size=20, weight='bold')


plt.tight_layout()
plt.show()
