from PymoNNto import *
import random as rnd
#rnd.seed(1)

class TextActivator(Behavior):

    def initialize(self, neurons):
        self.TextGenerator = neurons.TextGenerator
        #self.strength = self.parameter('strength', 1, neurons)

    def iteration(self, neurons):
        neurons.input_grammar = (neurons.y == neurons.current_char_index)#*self.strength
        neurons.voltage += neurons.input_grammar.astype(neurons.def_dtype)


class TextReconstructor(Behavior):

    def clear_history(self):
        self.reconstruction_history = ''

    def initialize(self, neurons):
        self.add_tag('Reconstructor')
        neurons.current_reconstruction_char = ''
        neurons.current_reconstruction_char_index = ''
        self.clear_history()

    def iteration(self, neurons):
        if neurons.TextGenerator is not None:

            neurons.rec_act = neurons.vector()
            for s in neurons.efferent_synapses['GLU']:
                if neurons.network.transposed_synapse_matrix_mode:  ######################################################
                    s.src.rec_act += s.W.dot(s.dst.spike)
                else:
                    s.src.rec_act += s.W.T.dot(s.dst.spike)

            if np.sum(neurons.rec_act)==0:
                neurons.current_reconstruction_char_index = -1
            else:
                index_act = np.sum(neurons.rec_act.reshape((neurons.height, neurons.width)), axis=1)
                neurons.current_reconstruction_char_index = np.argmax(index_act)

            neurons.current_reconstruction_char = neurons.TextGenerator.index_to_char(neurons.current_reconstruction_char_index)

            self.reconstruction_history += neurons.current_reconstruction_char


class TextReconstructor_ML(Behavior):##add "TextReconstructor" tag to constructor

    def act_to_indx(self, act):
        if np.sum(act) == 0:
            return -1
        else:
            return np.argmax(act)

    def get_char_index(self, pos):
        act = self.recon_act_buffer[pos]
        return self.act_to_indx(act)

    def get_char_indexes(self, start=0, end=-1):
        result = []
        for act in self.recon_act_buffer[start:end]:
            result.append(self.act_to_indx(act))
        return result

    @property
    def reconstruction_history(self):
        result=''
        for index in self.get_char_indexes(0, -1):
            result+=self.TextGenerator.index_to_char(index)
        return result

    def clear_history(self):
        self.recon_act_buffer = []

    def initialize(self, neurons):
        self.TextActivator = neurons.TextActivator
        self.TextGenerator = neurons.TextGenerator
        self.steps = 5

        self.clear_history()

        for n in neurons.network.NeuronGroups:
            n._rc_buffer_last = n.vector()
            n._rc_buffer_current = n.vector()

    def iteration(self, neurons):
        if self.TextActivator is not None:
            self.recon_act_buffer.append(0)

            #get current state
            for n in neurons.network.NeuronGroups:
                n._rc_buffer_last = n.spike

            result = []
            for step in range(self.steps):
                if step<len(self.recon_act_buffer):
                    #propagation
                    for s in neurons.network.SynapseGroups:
                        if 'GLU' in s.tags:
                            if neurons.network.transposed_synapse_matrix_mode:  ######################################################
                                s.src._rc_buffer_current = s.W.dot(s.dst._rc_buffer_last)  # +=
                            else:
                                s.src._rc_buffer_current = s.W.T.dot(s.dst._rc_buffer_last)  # +=


                    #collection
                    for n in neurons.network.NeuronGroups:
                        n._rc_buffer_last = n._rc_buffer_current.copy()

                    index_act = np.sum(neurons._rc_buffer_last.reshape((neurons.height, neurons.width)), axis=1)#get "activations" for different characters

                    self.recon_act_buffer[-(step+1)] += index_act


class TextGenerator(Behavior):

    initialize_on_init = True

    def get_text_blocks(self):
        return self.parameter('text_blocks', [])

    def unique(self, l):
        return list(sorted(set(l)))

    def get_str_history(self):
        return self.history

    def initialize(self, neurons):
        self.add_tag('Generator')

        self.text_blocks = self.get_text_blocks()
        self.current_block_index = -1
        self.position_in_current_block = -1
        self.alphabet = self.unique(''.join(self.text_blocks).replace('#','')) #list not string!!!!
        self.history = ''

        # output
        neurons.current_char = ''
        neurons.current_char_index = -1

        # manual activation
        self.next_char = None

        #if self.parameter('set_network_size_to_alphabet_size', False):
        #    dim = get_squared_dim(len(self.alphabet)) #NeuronDimension
        #    dim.initialize(neurons) #set size and x, y, z, width, height, depth

        char_count_vec = self.count_chars_in_blocks()
        self.char_weighting = char_count_vec / np.mean(char_count_vec)

        self.iterations_per_char = self.parameter('iterations_per_char', 1)

        #create aaabbbccc text
        for i in range(len(self.text_blocks)):
            new = ''
            for c in self.text_blocks[i]:
                for _ in range(self.iterations_per_char):
                    new += c
            self.text_blocks[i] = new

    def iteration(self, neurons):

        #if neurons.iteration%self.iterations_per_char==0 or neurons.current_char_index==-1:

        if self.next_char is not None:# manual activation
            neurons.current_char = self.next_char
            self.next_char = None
        else:
            neurons.current_char = self.get_char(next=True)

        neurons.current_char_index = self.char_to_index(neurons.current_char)

        self.history += neurons.current_char

    def get_char(self, next=False):
        if next:
            self.position_in_current_block += 1

        if self.position_in_current_block >= len(self.text_blocks[self.current_block_index]):
            self.current_block_index=self.get_next_block_index()
            self.position_in_current_block = 0

        return self.text_blocks[self.current_block_index][self.position_in_current_block]

    def index_to_char(self, index):
        if index>=0:
            return self.alphabet[index]
        else:
            return '#'

    def char_to_index(self, char):
        if char!='#':
            return self.alphabet.index(char)
        else:
            return -1



    def get_next_block_index(self):
        return rnd.randint(0, len(self.text_blocks)-1)
        #return np.random.randint(len(self.text_blocks))

    def set_next_char(self, char):#manual activation
        self.next_char = char

    def get_words(self):
        return unique([word for word in ' '.join(self.text_blocks).replace('.', ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ').split(' ') if word != ''])

    def count_chars_in_blocks(self):
        result = np.zeros(len(self.alphabet))
        for block in self.text_blocks:
            for c in block:
                result[self.char_to_index(c)] += 1
        return result

    def get_text_score(self, text):
        block_scores = [1 for _ in self.text_blocks]
        for i in range(len(text)):
            for bi, block in enumerate(self.text_blocks):
                block_score = 0
                comp_text = text[i:i+len(block)]
                comp_block = block[0:len(comp_text)]
                for t, b in zip(comp_text, comp_block):
                    if t == b:
                        block_score += 1 * (1/self.char_weighting[self.char_to_index(t)])

                block_scores[bi] += (block_score*block_score)/len(text)
        score = 0
        for bs in block_scores:
            score += np.sqrt(bs)
        return score

    def plot_char_distribution(self):
        import matplotlib.pyplot as plt

        cw=-np.sort(-self.char_weighting)

        print(cw)

        plt.barh(np.arange(len(cw)), cw)
        #plt.barh

        #self.char_weighting
        plt.show()



#########################
#Token_old
#########################

import tiktoken
import itertools


class TokenTextReconstructor(Behavior):

    @property
    def reconstruction_history(self):
        return ''.join([self.generator.index_to_char(token) for token in self.history])

    def clear_history(self):
        self.history = []

    def initialize(self, neurons):
        self.add_tag('Reconstructor')
        self.add_tag('TextReconstructor')
        neurons.current_reconstruction_char_index = -1
        neurons.current_reconstruction_char = '#'
        self.clear_history()
        self.generator = neurons.Generator

    def iteration(self, neurons):

        neurons.rec_act = neurons.vector()
        for s in neurons.efferent_synapses['GLU']:
            if neurons.network.transposed_synapse_matrix_mode:
                s.src.rec_act += s.W.dot(s.dst.spike)
            else:
                s.src.rec_act += s.W.T.dot(s.dst.spike)

        if np.sum(neurons.rec_act) == 0:
            neurons.current_reconstruction_char_index = -1
            neurons.current_reconstruction_char = '#'
        else:
            index_act = np.sum(neurons.rec_act.reshape((neurons.height, neurons.width)), axis=1)
            neurons.current_reconstruction_char_index = np.argmax(index_act)

        neurons.current_reconstruction_char = neurons.Generator.index_to_char(neurons.current_reconstruction_char_index)
        #print(neurons.current_reconstruction_char)

        self.history.append(neurons.current_reconstruction_char_index)


class TokenTextGenerator(Behavior):

    initialize_on_init = True

    #def unique(self, l):
    #    return list(sorted(set(l)))

    def tokenize(self, text_blocks):

        #str to tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        token_blocks = [encoding.encode(block) for block in text_blocks]
        all = list(itertools.chain.from_iterable(token_blocks))
        unique_tokens = np.unique(all)

        #tiktoken to dense index
        token_mapping = {t: i for i, t in enumerate(unique_tokens)}
        tokenized_text_blocks = [list(map(token_mapping.get, block, block)) for block in token_blocks]

        alphabet = [encoding.decode([t]) for t in unique_tokens]

        return tokenized_text_blocks, alphabet


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_tag('Generator')
        self.add_tag('TextGenerator')

        self.current_block_index = -1
        self.position_in_current_block = -1

        self.text_blocks = self.parameter('text_blocks', [])#[block.lower() for block in ]

        self.tokenized_text_blocks, self.alphabet = self.tokenize(self.text_blocks)

        self.history = []


        self.n_tokens = len(list(itertools.chain.from_iterable(self.tokenized_text_blocks)))
        self.n_unique_tokens = len(self.alphabet)

        temp = self.count_tokens_in_blocks()
        self.char_weighting = temp / np.mean(temp)
        print(self.char_weighting)
        self.count_chars_in_blocks = self.count_tokens_in_blocks

    def get_str_history(self):
        return "".join([self.index_to_char(c) for c in self.history])


    def count_tokens_in_blocks(self):
        result = np.zeros(len(self.alphabet))
        for block in self.tokenized_text_blocks:
            for t in block:
                result[t] += 1
        return result


    def initialize(self, neurons):
        neurons.current_char = ''
        neurons.current_char_index = -1


    def iteration(self, neurons):
        self.position_in_current_block += 1
        if self.position_in_current_block >= len(self.tokenized_text_blocks[self.current_block_index]):
            self.current_block_index = rnd.randint(0, len(self.tokenized_text_blocks)-1)
            self.position_in_current_block = 0

        neurons.current_char_index = self.tokenized_text_blocks[self.current_block_index][self.position_in_current_block]
        neurons.current_char = self.index_to_char(neurons.current_char_index)

        self.history.append(neurons.current_char_index)


    def index_to_char(self, index):
        if index>=0:
            return self.alphabet[index]
        else:
            return '#'

    def char_to_index(self, char):
        if char!='#':
            return self.alphabet.index(char)
        else:
            return -1

    #def get_text_score(self, text):
    #    return 0

    def get_text_score(self, token_list):
        block_scores = [1 for _ in self.text_blocks]
        for i in range(len(token_list)):
            for bi, block in enumerate(self.tokenized_text_blocks):
                block_score = 0
                comp_text = token_list[i:i+len(block)]
                comp_block = block[0:len(comp_text)]
                for t, b in zip(comp_text, comp_block):
                    if t == b:
                        block_score += 1 * (1/self.char_weighting[t])

                block_scores[bi] += (block_score*block_score)/len(token_list)
        score = 0
        for bs in block_scores:
            score += np.sqrt(bs)
        return score

    '''
    def count_chars_in_blocks(self):
        result = np.zeros(len(self.alphabet))
        for block in self.text_blocks:
            for c in block:
                result[self.char_to_index(c)] += 1
        return result



    def plot_char_distribution(self):
        import matplotlib.pyplot as plt

        cw=-np.sort(-self.char_weighting)

        print(cw)

        plt.barh(np.arange(len(cw)), cw)
        #plt.barh

        #self.char_weighting
        plt.show()
    '''


