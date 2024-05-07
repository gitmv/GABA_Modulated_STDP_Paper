from PymoNNto import *
from Behavior_Core_Modules import *
from Behavior_Text_Modules import *
from Helper import *
from Helper_Execution import *

ui = True
n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/10

grammar = get_random_sentences(3)
target_act = 1/n_chars(grammar)

net = Network(tag=ex_file_name(), settings=settings)

NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=green, behavior={
    # text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),

    # group output
    50: Output_TextActivator(),

    # text reconstruction
    80: TextReconstructor()
})

NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(n_exc_neurons), color=blue, behavior={
    # weight normalization
    3: Normalization(direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=200),
    3.1: Normalization(direction='afferent', syn_type='SOMA', exec_every_x_step=200),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=0.008735764741458582, init_sensitivity=0),

    # learning
    40: GABAModulation(transmitter='GABA', strength=6.450234496564654, avg_inh=0.3427857658747104, min=-0.15),
    41: STDP(transmitter='GLU', strength=0.0030597477411211885),

    # group output
    51: Output_Excitatory(exp=0.7378726012049153, mul=2.353594052973287),
})

NeuronGroup(net=net, tag='inh_neurons1', size=getGrid(n_inh_neuros), color=red, behavior={
    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # group output
    70: Output_Inhibitory(avg_inh=0.3427857658747104, target_activity=target_act, duration=2),
})

SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1', behavior={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='EE,GLU,DISTAL', src='exc_neurons1', dst='exc_neurons1', behavior={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons1', dst='inh_neurons1', behavior={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons1', dst='exc_neurons1', behavior={
    1: CreateWeights()
})

sm = StorageManager(net.tag, random_nr=True)
sm.backup_execued_file()

net.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui and not is_evolution_mode():
    from Helper_UI import *
    show_UI(net, sm)
else:
    #from Plots.DIST import *
    #train_and_render_dist_plot(net, input_steps=60000)

    # activate or deactivate for "no input" and/or "no GM" experiments
    #net.deactivate_behaviors('TextGenerator')
    #net.deactivate_behaviors('GABAModulation')
    add_recorders(net)
    train_and_generate_text(net, input_steps=60000, recovery_steps=10000, free_steps=5000, sm=sm)
    save_recorders(net, sm)
