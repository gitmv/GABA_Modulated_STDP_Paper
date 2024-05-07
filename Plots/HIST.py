import matplotlib.pyplot as plt
import numpy as np
from Plot_Helper import *



fig = plt.figure()

ax = fig.add_subplot(111, label="1")
ax2 = fig.add_subplot(111, label="2", frame_on=False)

fig.set_figwidth(6)
fig.set_figheight(6)

#ax = fig.add_subplot(1, 1, 1)



ax.spines[['right', 'top', 'left']].set_visible(False)
#ax.spines[['top']].set_visible(False)
ax.set_xticks([-1,0,1])
ax.set_yticks([0, 100, 200, 300])

ax.set_xlabel('voltage')

ax.set_xlim([-1, 1])
ax.set_ylim([0, 350])
#ax.set_ylim([-2.2, 2.7])

#ax.text(1.2, 0.0, 'voltage', c=(0, 0, 0, 1), size=10)#, ha='center'
#ax.text(0.0, 40.4*10, 'neurons in bin', c=(0, 0, 0, 1), size=10, ha='center')
#ax.text(0.0, 40.4, '  #', c=(1, 0, 1, 1), size=10, ha='right')
#ax.text(0.0, 40.4, '#  ', c=(0, 1, 1, 1), size=10, ha='right')
#ax.text(0.0, 42.4*10, 'spike probability', c=colorE, size=10, ha='center')

#ax.plot([-0.02, 0.02], [300, 300], linewidth=1, c='black')
#ax.text(0.04, 300, '300', va='center')

data = np.load('../Data/Char/E__voltage.npy')

voltages_untrained = data[1000]
voltages_trained = data[70000]

ax.hist(voltages_untrained, bins=100, color=color2, label='untrained', alpha=0.7)
ax.hist(voltages_trained, bins=100, color=color3, label='trained', alpha=0.7)

#ax.hist(np.random.normal(-1.5, 0.1, 1000), bins=100, color=color2, label='trained')#(1, 0, 1, 1)

ax.plot([0, 0], [0, 400], linewidth=0.8, c=(0, 0, 0, 0.1))


#ax.hist(np.random.normal(0.8, 0.1, 100), bins=100, color=color2)
#ax.hist(np.random.normal(0.8, 0.1, 100), bins=100, color=color2)


#ax.hist(np.random.normal(-0.4, 0.1, 1000), bins=100, color=color3, label='untrained')
ax.yaxis.set_label_position('left')
ax.set_ylabel('neurons in bin')



ax2.set_xlim([-1, 1])
ax2.set_ylim([0, 4])

ax2.set_xticks([])
ax2.set_yticks([0, 1])

ax2.yaxis.set_label_position('right')
ax2.set_ylabel('spike probability')
ax2.yaxis.tick_right()

ax2.yaxis.get_label().set_color(colorE)
ax2.yaxis.set_tick_params(labelcolor=colorE)  # Change color of tick labels
ax2.yaxis.set_tick_params(color=colorE)

x = np.arange(-2.0, 1.0, 0.001)
y = F_E(x)#*100#np.clip(np.power(np.clip(x, 0, None) * 2.353594052973287, 0.7378726012049153), 0.0, 1.0) * 10.0
ax2.plot(x, y, c=colorE)

ax.legend(loc=2)

#plt.text(-1.2, 42.1*10, 'B', size=20, weight='bold')
plt.text(-1.2, 42.1*10, 'B', size=20, weight='bold')

plt.tight_layout()
plt.show()

