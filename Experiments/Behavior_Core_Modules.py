from PymoNNto import *

settings = {'dtype': float32, 'syn_dim': SxD}

class Output(Behavior):

    def _initialize(self, neurons):  # overwrite
        return
    def initialize(self, neurons):
        self._initialize(neurons)
        neurons.voltage = neurons.vector()
        neurons.spike = neurons.vector(bool)
        neurons.spike_old = neurons.vector(bool)

    def _iteration(self, neurons):  # overwrite
        return
    def iteration(self, neurons):
        self._iteration(neurons)
        neurons._voltage = neurons.voltage.copy()  # for plotting and readout
        neurons.voltage.fill(0)


class Output_LineActivator(Output):

    def iteration(self, neurons):
        neurons.spike_old = neurons.spike.copy()
        if neurons.LineGenerator.behavior_enabled:
            neurons.input_lines = neurons.line_pic > 0
            neurons.spike = neurons.input_lines  # cast to bool when strength is active
        else:
            neurons.input_lines = neurons.vector('bool')
            neurons.spike = neurons.vector('bool')

class Output_TextActivator(Output):
    def _initialize(self, neurons):
        self.add_tag('TextActivator')

    def iteration(self, neurons):
        neurons.spike_old = neurons.spike.copy()
        if neurons.Generator.behavior_enabled:
            neurons.input_grammar = (neurons.y == neurons.current_char_index)  # *self.strength
            neurons.spike = neurons.input_grammar  # cast to bool when strength is active
        else:
            neurons.input_grammar = neurons.vector('bool')
            neurons.spike = neurons.vector('bool')


class Output_Inhibitory(Output):

    def _initialize(self, neurons):
        self.duration = self.parameter('duration', 2.0)
        self.durationPlus1 = self.duration + 1
        self.avg_inh = self.parameter('avg_inh', 0.28)
        self.target_activity = self.parameter('target_activity', 0.02)
        self.avg_act = 0

    def _iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.voltage) / (self.durationPlus1)
        neurons.spike = (self.avg_act * self.avg_inh / self.target_activity) > neurons.vector('uniform')


class Output_Excitatory(Output):

    def _initialize(self, neurons):
        self.mul = self.parameter('mul', 2.127)
        self.exp = self.parameter('exp', 2.127)

    def _iteration(self, neurons):
        neurons.spike_old = neurons.spike.copy()
        neurons.spike = np.power(np.clip(neurons.voltage * self.mul, 0.0, None), self.exp) > neurons.vector("uniform")


class SynapseOperation(Behavior):

    def initialize(self, neurons):
        self.transmitter = self.parameter('transmitter', None)
        self.strength = self.parameter('strength', 1.0)  # 1 or -1
        self.input_tag = 'input_' + self.transmitter
        self.init_call = compile('neurons.'+self.input_tag+'=neurons.vector()', '<string>', 'exec')
        self.set_call = compile('s.dst.'+self.input_tag+'+=s.add', '<string>', 'exec')
        exec(self.init_call)
        # setattr(neurons, self.input_tag, neurons.vector())

    def iteration(self, neurons):
        # setattr(neurons, self.input_tag, neurons.vector())
        exec(self.init_call)
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = np.sum(s.W[s.src.spike], axis=0) * self.strength
            s.dst.voltage += s.add
            exec(self.set_call)
            # setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)


class GABAModulation(Behavior):

    def initialize(self, neurons):
        self.strength = self.parameter('strength', 1.0)
        self.avg_inh = self.parameter('avg_inh', 0.28)
        self.min = self.parameter('min', 0.0)
        self.max = self.parameter('max', 1.0)
        self.input_tag = 'input_' + self.parameter('transmitter', 'GABA')
        self.get_call = compile('neurons.' + self.input_tag, '<string>', 'eval')

    def iteration(self, neurons):
        # inhibition = np.abs(getattr(neurons, self.input_tag))
        inhibition = eval(self.get_call)  # warning: inhibition value is negative: following sign switched from - to +!!
        neurons.li_stdp_mul = np.clip((1 + inhibition / self.avg_inh) * self.strength, self.min, self.max)


class IntrinsicPlasticity(Behavior):

    def initialize(self, neurons):
        self.strength = self.parameter('strength', 0.01)
        neurons.target_activity = self.parameter('target_activity', 0.02)
        neurons.sensitivity = neurons.vector()+self.parameter('init_sensitivity', 0.0)

    def iteration(self, neurons):
        neurons.sensitivity -= (neurons.spike - neurons.target_activity) * self.strength
        neurons.voltage += neurons.sensitivity


class STDP(Behavior):

    def initialize(self, neurons):
        self.transmitter = self.parameter('transmitter', None)
        self.eta_stdp = self.parameter('strength', 0.005)
        neurons.li_stdp_mul = neurons.vector(1)

    def iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            src_len = np.sum(s.src.spike_old)
            weight_change = s.dst.li_stdp_mul[s.dst.spike] * self.eta_stdp
            dw = np.tile(weight_change, (src_len, 1))

            mask = np.ix_(s.src.spike_old, s.dst.spike)
            s.W[mask] += dw

            #if neurons.GABAModulation.min < 0:
            s.W[mask] = np.clip(s.W[mask], 0.0, None)


class Normalization(Behavior):

    def initialize(self, neurons):
        self.syn_type = self.parameter('syn_type', 'GLU')
        self.exec_every_x_step = self.parameter('exec_every_x_step', 1)
        self.afferent = 'afferent' in self.parameter('direction', 'afferent')
        self.efferent = 'efferent' in self.parameter('direction', 'afferent')

    def iteration(self, neurons):
        if (neurons.iteration-1) % self.exec_every_x_step == 0:
            if self.afferent:
                self.norm(neurons, neurons.afferent_synapses[self.syn_type], axis=0)
            if self.efferent:
                self.norm(neurons, neurons.efferent_synapses[self.syn_type], axis=1)

    def norm(self, neurons, syn, axis):
        neurons._temp_ws = neurons.vector()
        for s in syn:
            neurons._temp_ws += np.sum(s.W, axis=axis)
        neurons._temp_ws[neurons._temp_ws == 0.0] = 1.0  # avoid division by zero error
        for s in syn:
            s.W /= (neurons._temp_ws[:, None] if axis == 1 else neurons._temp_ws)


class CreateWeights(Behavior):

    def initialize(self, synapses):
        distribution = self.parameter('distribution', 'uniform(0.0,1.0)')  # ones
        density = self.parameter('density', 1.0)

        synapses.W = synapses.matrix(distribution, density=density) * synapses.enabled

        #  self.remove_autapses = self.parameter('remove_autapses', False) and synapses.src == synapses.dst

        if self.parameter('normalize', True):
            for i in range(10):
                synapses.W /= np.sum(synapses.W, axis=1)[:, None]
                synapses.W /= np.sum(synapses.W, axis=0)

            synapses.W *= self.parameter('nomr_fac', 1.0)


    def iteration(self, synapses):
        return
        #  if self.remove_autapses:
        #    np.fill_diagonal(synapses.W, 0.0)

