#import matplotlib
#matplotlib.use('ps')
#from matplotlib import rc
#rc('text', usetex=True)
#rc('text.latex', preamble='\\usepackage{color}')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from Plot_Helper import *
from matplotlib.patches import Rectangle

ticklabelpad = mpl.rcParams['xtick.major.pad']

fig, ax = plt.subplots(6, 1)

fig.set_figwidth(8)
fig.set_figheight(9)


def txt_line_break(txt, max_l):
    txt=txt.replace('\n', ' ')
    counter=0
    result = ''
    for s in txt:
        result += s
        counter+=1
        if counter > max_l and (s==' ' or s=='.' or s==','):
            result += '\n'
            counter = 0
    print('(', result[0], ')')

    return result


txt = [
    ('No Input', 'No GABA Mod.'),
    ('No Input', 'GABA Mod.'),
    ('Char. Text Input', 'No GABA Mod.'),
    #('Bar Input', 'No GABA Mod.'),
    ('Char. Text Input', 'GABA Mod.'),
    ('Bar Input', 'GABA Mod.'),
    ('Token Text Input', 'GABA Mod.')]

#input_txt = [('', '', ''),
#    ('', '', ''),
#    ('...fox eats meat. boy...', '...sdfasfh...', '...sdfasfh...'),
#    ('', '', ''),
#    ('...fox eats meat. boy...', '...sdfasfh...', '...boy drinks juice. pengu... *'),
#    ('', '', ''),
#    ('...Cortical networks are capable of...', '...signals here GABA...', '...called GABA-Modulated STDP... *')]

folders = [
    '../Data/No_Input_No_GM/',
    '../Data/No_Input_With_GM/',
    '../Data/Char_No_GM/',
    #'../Data/Bar_No_GM/',
    '../Data/Char/',
    '../Data/Bar/',
    '../Data/Token/'
    ]

char = ['A',
        'B',
        'C',
        'D',
        'E',
        'F'
        ]#,'G'

y_scale = 2

l = 1000
s = 500

x = np.array(range(0, l))/s*5
xs = x[0:s]

#print(x)

for i, (a, t, f, ch) in enumerate(zip(ax, txt, folders, char)):


    #if i == 0 or i == 1:
    #    cp1 = colorP3
    #    cp2 = colorP3
    #else:
    cp1 = colorP1
    cp2 = colorP2

    #if sr is None:
    #    sr = (np.sin(np.arange(0, 75000, 1)/10 / (1+1*i))+1)*0.5*0.2

    sr = np.load(f+'E_np.mean(spike).npy')

    if sr is not None:
        end = len(sr)
        start = 0
        cut_off = end-5000-10000
        free = end-5000

        a.plot(x, sr[start:l]*y_scale, c=cp1, linewidth=1)
        a.plot(xs+10+2, sr[cut_off-s:cut_off]*y_scale, c=cp1, linewidth=1)

        a.plot([15+2, 15+2], [sr[cut_off-1]* y_scale, sr[cut_off]* y_scale], c=cp1, linewidth=1)

        a.plot(xs+15+2, sr[cut_off:cut_off+s]*y_scale, c=cp2, linewidth=1)
        #a.plot(x+11+5+1, sr[free-l:free]*y_scale, c=colorP2, linewidth=1)

        #a.plot(x+22, sr[free:free+l]*y_scale, c=colorP3, linewidth=1)
        a.plot(x+22+2, sr[end-l:end]*y_scale, c=colorP3, linewidth=1)

    a.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    #a.spines['left'].set_color('lightgray')
    #a.spines['bottom'].set_color('lightgray')

    y_offset = -0.01

    a.plot([0, 0], [-0.4, 1], linewidth=1, c='black')
    #if i==6:
    color = 'black'
    #else:
    #    color = 'gray'
    a.plot([0, 10], [y_offset, y_offset], linewidth=1, c=color)
    a.plot([10, 12], [y_offset, y_offset], linewidth=1, c=color, linestyle='dotted')
    a.plot([12, 22], [y_offset, y_offset], linewidth=1, c=color)
    a.plot([22, 24], [y_offset, y_offset], linewidth=1, c=color, linestyle='dotted')
    a.plot([24, 34], [y_offset, y_offset], linewidth=1, c=color)

    if i==0:
        a.plot([5,5],[-0.04,0.02], c='black', linewidth=1)
        a.text(5,-0.2,'500', ha='center', size=10)

        a.text(34.2, 0, 't', va='center')


    a.set_yticks([])#1.5
    a.set_xticks([])
    a.set_ylim([-0.4, 1])#y_offset*2
    a.set_xlim([-0.1, 34])

    if i==0:
        a.text(5.5, .9, 'Input Phase', ha='center', size=10, c=colorP1)
        a.text(17, .9, 'Recovery Phase', ha='center', size=10, c=colorP2)
        a.text(29, .9, 'Free Phase', ha='center', size=10, c=colorP3)

    if i == 0:
        a.set_ylabel("spike rate")

    if i == 5:
        #a.spines[['bottom']].set_visible(True)
        #a.spines['bottom'].set_color('black')

        #a.annotate('t', xy=(1, 0), xytext=(5, -ticklabelpad), ha='left', va='top', xycoords='axes fraction', textcoords='offset points')

        #a.set_xticks([0, 5, 17], [0, s, 'input cut off'])
        a.set_xticks([])


    if i == 2 or i == 3 or i == 5:
        cn = 11

        #if i == 6:
        #    cn=8

        inp_data = np.load(f + 'S_current_char.npy')
        recon_data = np.load(f + 'S_current_reconstruction_char.npy')


        txt_input = "".join(inp_data[500-cn:500+cn])
        txt_cut_B = "".join(inp_data[cut_off-cn:cut_off])
        txt_cut_A = "".join(recon_data[cut_off+1:cut_off+cn+1])
        txt_recon = "".join(recon_data[-500-cn:-500+cn])

        if i == 2:
            txt_cut_A = txt_cut_A[:-3]
            txt_recon = txt_recon[:-2]

        if i == 5:
            a.text(5, -0.3 -0.65 + 0.02, txt_line_break(txt_input, 18)[:-2], ha='center', size=8, c=colorP1)
            a.text(17, -0.3 -0.65 + 0.46 - 0.12, txt_line_break(txt_cut_B, 18), ha='center', size=8, c=colorP1)
            a.text(17, -0.3 -0.65, txt_line_break(txt_cut_A, 18), ha='center', size=8, c=colorP2)
            a.text(29, -0.3 -0.65 + 0.18, txt_line_break(txt_recon, 18), ha='center', size=8, c=colorP3)
        else:
            a.text(5, -0.3, txt_input, ha='center', size=8, c=colorP1)
            a.text(17, -0.3 , txt_cut_B, ha='right', size=8, c=colorP1)
            a.text(17, -0.3, txt_cut_A, ha='left', size=8, c=colorP2)
            a.text(29, -0.3, txt_recon, ha='center', size=8, c=colorP3)


        #plt.plot([17, 17], [-0.8,-0.1], linewidth=1, c='black')

        #a.text(5, -0.3+add, txt_input, ha='center', size=7, c=colorP1)
        #a.text(17, -0.3+add+add2, txt_cut_B, ha='center', size=7, c=colorP1)
        #a.text(17, -0.3+add, txt_cut_A, ha='center', size=7, c=colorP2)
        #a.text(29, -0.3+add, txt_recon, ha='center', size=7, c=colorP3)


    #def mark_area()

    if i == 2 or i == 3 or i == 4 or i == 5 or i == 6:
    #if i == 4:
        y_m = -0.05
        w = x[cn]#0.3

        if i==4:#i == 3 or i == 5:
            w=x[3]

        a.plot([5-w, 5+w], [y_m, y_m], linewidth=1, c='black')
        #a.plot([5 - w, 5 - w], [y_m-0.01, y_m+0.015], linewidth=1, c='black')
        #a.plot([5 + w, 5 + w], [y_m-0.01, y_m+0.015], linewidth=1, c='black')
        a.plot([17-w, 17+w], [y_m, y_m], linewidth=1, c='black')
        a.plot([29-w, 29+w], [y_m, y_m], linewidth=1, c='black')

    for j, t in enumerate(t):
        if 'No' in t:
            c = red
        else:
            c = 'black'
        text = a.text(-9, -j/6+0.55-0.25, t, ha='left', size=10, c=c)

    a.text(-11, 0.42-0.25, ch, ha='left', size=20)





    #img = image.imread('../Data/bar2.png')

    size = 0.03

    if i == 4:#i == 3 or

        inp_data = np.load(f +'S_input_lines.npy')
        recon_data = np.load(f +'S_rec_pic.npy')

        #if i==3:
        #    yp = 0.456
        #else:
        #yp = 0.186
        yp = 0.216

        jm = 0.0331


        for j in range(6):

            pic = inp_data[500-3+j].reshape(30,30)

            axicon = fig.add_axes([0.2734 + j*jm, yp, size, size])  # + 0.5+0*3 b.ymin #0.861
            axicon.imshow(pic, interpolation='bilinear', cmap='gray')
            axicon.set_xticks([])
            axicon.set_yticks([])
            axicon.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

        xp = 0.5215

        for j in range(3):

            pic = inp_data[60000-3+j].reshape(30, 30)

            axicon = fig.add_axes([xp + j*jm, yp, size, size])  # + 0.5+0*3 b.ymin
            axicon.imshow(pic, interpolation='bilinear', cmap='gray')
            axicon.set_xticks([])
            axicon.set_yticks([])
            axicon.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

        for j in range(3):

            pic = recon_data[60000+j+1].reshape(30, 30)

            axicon = fig.add_axes([xp + (j+3)*jm, yp, size, size])  # + 0.5+0*3 b.ymin
            axicon.imshow(pic, interpolation='bilinear', cmap='gray')
            axicon.set_xticks([])
            axicon.set_yticks([])
            axicon.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

        for j in range(6):

            pic = recon_data[-500-3+j].reshape(30, 30)

            axicon = fig.add_axes([0.7685 + j*jm, yp, size, size])  # + 0.5+0*3 b.ymin
            axicon.imshow(pic, interpolation='bilinear', cmap='gray')
            axicon.set_xticks([])
            axicon.set_yticks([])
            axicon.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

        '''
        xp = 0.29#0.395
        axicon = fig.add_axes([xp+0+0.06*0, 1-0.02-0.05-((1-0.035)/7)*i, size, size])#+ 0.5+0*3 b.ymin
        axicon.imshow(img, interpolation='bilinear')
        axicon.set_xticks([])
        axicon.set_yticks([])

        axicon = fig.add_axes([xp+0+0.06*1, 1-0.02-0.05-((1-0.035)/7)*i, size, size])#+ 0.5+0*3 b.ymin
        axicon.imshow(img, interpolation='bilinear')
        axicon.set_xticks([])
        axicon.set_yticks([])

        axicon = fig.add_axes([xp+0+0.06*2, 1-0.02-0.05-((1-0.035)/7)*i, size, size])#+ 0.5+0*3 b.ymin
        axicon.imshow(img, interpolation='bilinear')
        axicon.set_xticks([])
        axicon.set_yticks([])



        axicon = fig.add_axes([xp+0.235+0.06*1, 1-0.02-0.05-((1-0.035)/7)*i, size, size])#+ 0.5+0*3 b.ymin
        axicon.imshow(img, interpolation='bilinear')
        axicon.set_xticks([])
        axicon.set_yticks([])



        axicon = fig.add_axes([xp+0.47+0.06*0, 1-0.02-0.05-((1-0.035)/7)*i, size, size])#+ 0.5+0*3 b.ymin
        axicon.imshow(img, interpolation='bilinear')
        axicon.set_xticks([])
        axicon.set_yticks([])

        axicon = fig.add_axes([xp+0.47+0.06*1, 1-0.02-0.05-((1-0.035)/7)*i, size, size])#+ 0.5+0*3 b.ymin
        axicon.imshow(img, interpolation='bilinear')
        axicon.set_xticks([])
        axicon.set_yticks([])

        axicon = fig.add_axes([xp+0.47+0.06*2, 1-0.02-0.05-((1-0.035)/7)*i, size, size])#+ 0.5+0*3 b.ymin
        axicon.imshow(img, interpolation='bilinear')
        axicon.set_xticks([])
        axicon.set_yticks([])
        '''

#ax = fig.gca()
#rect = Rectangle((-11.1, -1), 1, 1.5, linewidth=1, edgecolor='r', facecolor='orange', alpha=0.5,clip_on=False)
#rect = Rectangle((-1, -1), 1, 1, linewidth=1, edgecolor='r', facecolor='orange', alpha=0.5, clip_on=False)
#ax.add_patch(rect)

plt.tight_layout()
plt.show()

#plt.savefig('filename.png', dpi=600)
