from PymoNNto.Exploration.Evolution.Interface_Functions import *
from PymoNNto import *
from PymoNNto.Exploration.AnalysisModules import *

def add_recorders(net):
    if hasattr(net.inp_neurons, 'LineGenerator'):
        net.inp_neurons.add_behavior(100, Recorder(['spike', 'np.mean(spike)', 'input_lines', 'rec_pic'], tag='rec'))
    else:
        net.inp_neurons.add_behavior(100, Recorder(['spike', '_voltage', 'np.mean(spike)', 'np.mean(_voltage)', 'current_char', 'current_char_index', 'current_reconstruction_char_index', 'current_reconstruction_char'], tag='rec'))

    net.exc_neurons1.add_behavior(100, Recorder(['_voltage', 'spike', 'sensitivity', 'np.mean(_voltage)', 'np.mean(spike)', 'np.mean(sensitivity)'], tag='rec'))
    net.inh_neurons1.add_behavior(100, Recorder(['_voltage', 'spike', 'np.mean(_voltage)', 'np.mean(spike)'], tag='rec'))

def save_recorders(net, sm):
    sm.save_recorder('S_', net.inp_neurons.rec)
    sm.save_recorder('E_', net.exc_neurons1.rec)
    sm.save_recorder('I_', net.inh_neurons1.rec)

def train_and_generate_text(net, input_steps, recovery_steps, free_steps, sm=None):
    #net.deactivate_behaviors('Reconstructor')

    net.simulate_iterations(input_steps, 100)

    # deactivate Input
    #net.deactivate_behaviors('STDP')
    #net.deactivate_behaviors('Normalization')
    net.deactivate_behaviors('Generator')
    #net.activate_behaviors('Reconstructor')
    net.exc_neurons1.add_behavior(90, Recorder(variables=['np.mean(n.spike)'], tag='srec'))

    net.simulate_iterations(recovery_steps, 100)

    # text generation
    net.simulate_iterations(free_steps, 100)

    # scoring
    tr = net.inp_neurons.Reconstructor
    if 'char' in net.tag or 'token' in net.tag:
        txt = tr.reconstruction_history[-free_steps:]
        txt_score = net['Generator', 0].get_text_score(txt)
        osc_score = np.clip(np.mean(net.exc_neurons1.target_activity-np.abs(net.exc_neurons1.srec['np.mean(n.spike)', 0, 'np'] - net.exc_neurons1.target_activity))/net.exc_neurons1.target_activity, 0.00001, None)
        dist_score, classes = get_class_score(net)

        set_score(txt_score * osc_score * dist_score, info={
                'text': txt,
                'osc_score': osc_score,
                'txt_score': txt_score,
                'dist_score': dist_score,
                'classes': str(classes),
                'simulated_iterations': net.iteration
        }, sm=sm)





class Weight_Classifier_PreT(Classifier_base):

    def get_data_matrix(self, neurons):
        syn_tag = self.parameter('syn_tag', 'EE')
        return neurons.afferent_synapses[syn_tag][0].W #get_partitioned_synapse_matrix(neurons, syn_tag, 'W')

def get_class_score(net):
    if net['ES', 0] is None:
        return 1, []

    wcp = Weight_Classifier_PreT(net.exc_neurons1, syn_tag='ES')
    tg = net['Generator', 0]

    classification = wcp(sensitivity=3.0)
    classes = []
    for i in range(len(tg.alphabet)):
        classes.append(np.sum(np.equal(classification, i + 1)))
    classes = -np.sort(-np.array(classes))

    cw = -np.sort(-tg.char_weighting)
    cw = net.exc_neurons1.size / len(tg.alphabet) * cw
    score = 1.0 - np.sum(np.abs(classes - cw)) / net.exc_neurons1.size / 2.0

    #print(np.array2string(classes, separator=", "), net.iteration, score)  # repr: with commas

    return score, classes