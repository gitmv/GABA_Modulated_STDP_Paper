from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
from PymoNNto.Exploration.AnalysisModules import *

from PymoNNto.Exploration.Network_UI.Advanced_Tabs import *


def show_UI(net, sm, qa=['STDP', 'Normalization', 'GABAModulation', 'TextGenerator', 'LineGenerator'], additional_modules=None):

    # add classifier modules to neuron group
    if net['EE', 0] is not None:
        Weight_Classifier_Pre(net.exc_neurons1, syn_tag='EE')
    if net['ES', 0] is not None:
        Weight_Classifier_Pre(net.exc_neurons1, syn_tag='ES')

    # create ui tab dict
    my_modules = get_modules_dict(
        get_default_UI_modules(['spike'], quick_access_tags=qa),
        get_my_default_UI_modules(),
        additional_modules
    )

    #modify some tabs
    my_modules[multi_group_plot_tab].__init__(['spike|target_activity|0.0|target_activity*2', '_voltage', 'sensitivity', 'input_GABA*(-1)|LI_threshold', 'linh'])
    my_modules[single_group_plot_tab].__init__(['spike', '_voltage', 'input_GLU', 'input_GABA', 'input_grammar', 'sensitivity'], net_lines=[0.02], neuron_lines=[0, 0.5, 1.0])
    my_modules[reconstruction_tab].__init__(recon_groups_tag='exc_neurons1')

    #create classification AnalysisModules to classify characters and input-non-input neuron classification
    neurons = net.exc_neurons1

    if hasattr(net.inp_neurons, 'Output_TextActivator'):
        neurons = net.inp_neurons
        neurons.Input_Weights = np.zeros((neurons.size, len(net['TextGenerator', 0].alphabet)))
        neurons.Input_Weights[np.arange(neurons.size).astype(int), neurons.y.astype(int)] = 1

    if hasattr(neurons, 'Input_Weights'):
        char_classes = np.sum((neurons.Input_Weights>0) * np.arange(1, neurons.Input_Weights.shape[1]+1, 1), axis=1).transpose()#neurons.Input_Weights.shape[1]
        Static_Classification(parent=neurons, name='char', classes=char_classes)

    if hasattr(neurons, 'Input_Mask'):
        Static_Classification(parent=neurons, name='input class', classes=neurons.Input_Mask)

    # launch ui
    Network_UI(net, modules=my_modules, title=net.tags[0], storage_manager=sm, group_display_count=len(net.NeuronGroups), reduced_layout=False).show()



from PymoNNto.Exploration.Network_UI.TabBase import *
from PymoNNto.Exploration.Visualization.Visualization_Helper import *

class recon_tab(TabBase):

    def initialize(self, Network_UI):
        self.img_item = Network_UI.sidebar.add_plot().add_image()

    def update(self, Network_UI):
        group = Network_UI.network.inp_neurons
        if hasattr(group, 'rec_pic'):
            self.img_item.setImage(np.rot90(np.dstack([group.rec_pic, group.rec_pic, group.rec_pic]), 3))
