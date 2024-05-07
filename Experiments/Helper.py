import os
import sys

def ex_file_name():
    return os.path.basename(sys.argv[0]).replace('.py', '')

def n_unique_chars(grammar):
    return len(set(''.join(grammar).replace('#', '')))

def n_chars(grammar):
    return len(''.join(grammar))

def get_random_sentences(n_sentences):
    sentences = [' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' the fish swims.', ' plant loves rain.', ' parrots can fly.']
    return sentences[0:n_sentences]

def get_char_sequence(n_chars):
    sequence = '. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[](){}<>'
    return [sequence[0:n_chars]]

def get_long_text():
    return [' fox eats meat. boy drinks juice. penguin likes ice.']

def get_paper_abstract():
    return ["""
The Python Modular Neural Network Toolbox (PymoNNto) provides a versatile and adaptable Python-based framework to develop and investigate neural networks.
In contrast to other commonly used simulators such as Brian2 and NEST, PymoNNto imposes only minimal restrictions for implementation and execution.
The basic structure of PymoNNto consists of one network class with several neuron- and synapse-groups.
The behaviour of each group can be flexibly defined by exchangeable modules.
The implementation of these modules is up to the user and only limited by Python itself.
Behaviours can be implemented in Python, Numpy, Tensorflow and other libraries to perform computations on CPUs and GPUs.
"""]