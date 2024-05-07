import numpy as np
from PymoNNto import *
from PIL import Image, ImageDraw

class LineGenerator(Behavior):

    def initialize(self, neurons):
        self.add_tag('Generator')
        self.patterns = []
        self.counter = 0

        center_x = self.parameter('center_x', 1)
        center_y = self.parameter('center_y', 1)
        degree = np.deg2rad(self.parameter('degree', 1))
        line_length = self.parameter('line_length', 1)

        pattern_count=1
        for v in [center_x, center_y, degree]:
            if type(v) in [list, np.ndarray]: pattern_count = np.maximum(pattern_count, len(v))

        center_x *= np.ones(pattern_count)
        center_y *= np.ones(pattern_count)
        degree *= np.ones(pattern_count)

        self.grid_width = neurons.width
        self.grid_height = neurons.height

        for i in range(pattern_count):
            self.patterns.append(self.getLinePicture(degree[i], center_x[i], center_y[i], line_length/2, self.grid_width, self.grid_height))

    def iteration(self, neurons):
        self.counter += 1
        if self.counter >= len(self.patterns):
            self.counter = 0
        neurons.line_pic = self.patterns[self.counter]

    def pol2cart(self, theta, rho):
        return rho * np.cos(theta), rho * np.sin(theta)

    def cart2pol(self, x, y):
        return np.arctan2(y, x), np.hypot(x, y)

    def rotatearoundpoint(self, point=(0, 0), degree=0, center=(0, 0)):
        theta, rho = self.cart2pol(point[0] - center[0], point[1] - center[1])
        x, y = self.pol2cart(theta + degree, rho)
        return x + center[0], y + center[1]

    def getLinePicture(self, deg, center_x, center_y, length, width, height):
        im = Image.new('L', (width, height), (0))
        draw = ImageDraw.Draw(im)
        rot_point = self.rotatearoundpoint((length, 0), deg, (0, 0))
        draw.line((center_x - np.floor(rot_point[0]), center_y - np.floor(rot_point[1]),
                   center_x + np.floor(rot_point[0]), center_y + np.floor(rot_point[1])), fill=255)
        #im.show()
        return np.array(im.getdata())#picture_to_array(im).flatten()

class LineReconstructor(Behavior):

    def initialize(self, neurons):
        self.add_tag('Reconstructor')
        neurons.rec_pic = np.random.rand(neurons.width, neurons.height)#neurons.LineGenerator.grid_height, neurons.LineGenerator.grid_width

    def iteration(self, neurons):
        #if neurons.LineGenerator is not None:

        neurons.rec_act = neurons.vector()
        for s in neurons.efferent_synapses['GLU']:
            if neurons.network.transposed_synapse_matrix_mode:
                s.src.rec_act += s.W.dot(s.dst.spike)
            else:
                s.src.rec_act += s.W.T.dot(s.dst.spike)

        neurons.rec_pic = neurons.rec_act.reshape(neurons.width, neurons.height)#neurons.LineGenerator.grid_height, neurons.LineGenerator.grid_width


'''
class ImageGenerator(Behavior):

    def initialize(self, neurons):
        self.patterns = []
        self.counter = 0
        for file_name in self.parameter(files):
            image = Image.open(file_name)
            image_gray = pil_image.convert('L')
            self.patterns.append(np.array(pil_image_gray).flatten().astype(np.float32))

    def iteration(self, neurons):
        self.counter += 1
        if self.counter >= len(self.patterns):
            self.counter = 0
'''