import numpy as np
import matplotlib.pyplot as plt

#(1, 0.5, 0, 0.7)
#(0, 0.8, 0.8, 0.7)
#(0.5, 0, 1, 0.7)

color1 = (0, 176/255, 80/255, 1)#(0, 158/255, 132/255, 1)#Misc. 1
color2 = (1/255, 127/255, 157/255, 1)#(166/255, 76/255, 255/255, 1.0)#(0.5, 0, 1, 0.7)#(202/255, 0, 110/255, 1)#Misc. 2
color3 = (253/255, 97/255, 0, 1)#(0, 219/255, 219/255, 1.0)#(0, 0.8, 0.8, 0.7)#(250/255, 101/255, 0, 1)#Misc. 3

blue = (12/255, 5/255, 177/255, 1)#(0, 0, 1.0, 1)
red = (253/255, 18/255, 0, 1)#(1.0, 0, 0, 1)


colorS = color1#(153/255, 233/255, 0, 1)#source
colorE = blue#excitatory
colorI = red#Inhibitory

colorLTP = color1#(0, 176/255, 80/255, 1)#colorS#(0, 1.0, 0, 1)
colorLTD = red#(1.0, 0, 0, 1)

colorSTDP = color2#(0, 0, 1.0, 1)
#colorSTDP1 = (0, 0, 1.0, 0.7)#(.3, .3, 1, 1)
#colorSTDP2 = (0, 0, 1.0, 0.4)#(.6, .6, 1, 1)

colorCA = color3#(.6, 0, .6, 1)
#colorCA2 = (.6, 0, .6, 0.8)#(.8, .2, .8, 1)
#colorCA3 = (.6, 0, .6, 0.6)#(1, .4, 1, 1)


colorP1 = color1#colorLTP#colorS#color1
colorP2 = color3#(0.5, 0, 1, 0.7)#color2
colorP3 = color2#(0, 0.8, 0.8, 0.7)#color3



theta = 0.3427857658747104
target_act = 0.01923
gamma = 6.450234496564654
min_m = -0.15

def M(gaba):
    return np.clip((1 - gaba / theta) * gamma, min_m, 1.0)


def F_E(v):
    return np.clip(np.power(np.clip(v, 0, None) * 2.353594052973287, 0.7378726012049153), 0.0, 1.0)

def F_I(v):
    return np.clip((v * 0.34 / 0.01923), 0, 1)



def ca(x, g):
    return np.power(np.e, -np.power((5 * np.log(np.clip(-x + 2, 0.0001, None)) * 0.185), 2)) * (1 - g) * ((x+2)/4)

def ltp(ca2):
    return np.tanh(ca2)

def ltd(ca2):
    return np.tanh(ca2*6)/3

def stdp(ca2):
    return ltp(ca2)-ltd(ca2)