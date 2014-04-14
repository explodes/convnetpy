import math
import random
import time
from collections import namedtuple


REVISION = 'ALPHA'

return_v = False
v_val = True

# Random number utilities

def guass_random():
    global return_v
    global v_val
    if return_v:
        return_v = False
        return v_val
    u = 2 * random.random() - 1
    v = 2 * random.random() - 1
    r = u * u + v * v
    if r == 0 or r > 1:
        return guass_random()
    c = math.sqrt(-2 * math.log(r) / r)
    v_val = v * c  # cache this
    return_v = True
    return u * c


def randf(a, b):
    return (b - a) * random.random() + a


def randi(a, b):
    return math.floor((b - a) * random.random() + a)


def randn(mu, std):
    return mu + guass_random() * std


# Array utilities

Maxima = namedtuple("Maxima", ["maxi", "maxv", "mini", "minv", "dv"])


def maxim(w):
    """ return max and min of a given non-empty array. """
    if not w:
        return {}
    else:
        maxv = minv = w[0]
        maxi = mini = 0
        for index, item in enumerate(w[1:]):
            if item > maxv:
                maxi, maxv = index, item
            if item < minv:
                mini, minv = index, item
        return Maxima(maxi, maxv, mini, minv, maxv - minv)


def augment(V, crop, dx=None, dy=None, fliplr=False):
    """
    Volume utilities
    intended for use with data augmentation
    crop is the size of output
    dx,dy are offset wrt incoming volume, of the shift
    fliplr is boolean on whether we also want to flip left<->right
    """
    # note assumes square outputs of size crop x crop
    if dx is None:
        dx = randi(0, V.sx - crop)
    if dy is None:
        dy = randi(0, V.sy - crop)

    # randomly sample a crop in the input volume
    if crop != V.sx or dx != 0 or dy != 0:
        W = Vol(crop, crop, V.depth, 0.0)
        for x in xrange(crop):
            for y in xrange(crop):
                if x + dx < 0 or x + dx >= V.sx or y + dy >= V.sy:
                    # oob
                    continue
                for d in xrange(V.depth):
                    W.set(x, y, d, V.get(x + dx, y + dy, d))  # copy data over
    else:
        W = V

    # flip volume horizontally
    if fliplr:
        W2 = W.clone_and_zero()
        for x in xrange(W.sx):
            for y in xrange(W.sy):
                for d in xrange(W.depth):
                    W2.set(x, y, d, W.get(W.sx - x - 1, y, d))  # copy data over
        W = W2  # swap

    return W


def img_to_vol(img, convert_grayscale=False):
    """
    img is a PIL Image
    returns a Vol of size (W, H, 4). 4 is for RGBA
    """
    bands = img.getbands()
    if bands == ('R', 'G', 'B', 'A'):
        return rgba_to_vol(img, convert_grayscale=convert_grayscale)
    else:
        raise ValueError("Unsupported band format")


def rgba_to_vol(img, convert_grayscale=False):
    width, height = img.size()
    d = img.getdata()

    n = width * height * 4
    pv = [0] * n

    for index in xrange(0, n, 4):
        r, g, b, a = d[index / 4]
        pv[index] = r / 255.0 - 0.5
        pv[index + 1] = g / 255.0 - 0.5
        pv[index + 2] = b / 255.0 - 0.5
        pv[index + 3] = a / 255.0 - 0.5

    x = Vol(width, height, 4, 0.0)
    x.w = pv

    if convert_grayscale:
        x1 = Vol(width, height, 1, 0.0)
        for i in xrange(width):
            for j in xrange(height):
                x1.set(i, j, 0, x.get(i, j, 0))
        x = x1

    return x


class Vol(object):
    """
    Vol is the basic building block of all data in a net.
    it is essentially just a 3D volume of numbers, with a
    width (sx), height (sy), and depth (depth).
    it is used to hold data for all filters, all volumes,
    all weights, and also stores all gradients w.r.t. 
    the data. c is optionally a value to initialize the volume
    with. If c is missing, fills the Vol with random numbers.
    """

    def __init__(self, sx, sy, depth, c=None):
        self.sx = sx
        self.sy = sy
        self.depth = depth

        n = sx * sy * depth

        self.dw = [0] * n

        if c is None:
            # weight normalization is done to equalize the output
            # variance of every neuron, otherwise neurons with a lot
            # of incoming connections have outputs of larger variance
            self.w = [0] * n
            scale = math.sqrt(1.0 / n)
            for i in xrange(n):
                self.w[i] = randn(0.0, scale)
        else:
            self.w = [c] * n


    def get(self, x, y, d):
        index = ((self.sx * y) + x) * self.depth + d
        return self.w[index]

    def set(self, x, y, d, v):
        index = ((self.sx * y) + x) * self.depth + d
        self.w[index] = v

    def add(self, x, y, d, v):
        index = ((self.sx * y) + x) * self.depth + d
        self.w[index] += v

    def get_grad(self, x, y, d):
        index = ((self.sx * y) + x) * self.depth + d
        return self.dw[index]

    def set_grad(self, x, y, d, v):
        index = ((self.sx * y) + x) * self.depth + d
        self.dw[index] = v

    def add_grad(self, x, y, d, v):
        index = ((self.sx * y) + x) * self.depth + d
        self.dw[index] += v

    def clone_and_zero(self):
        return Vol(self.sx, self.sy, self.depth, 0.0)

    def clone(self):
        V = Vol(self.sx, self.sy, self.depth, 0.0)
        n = len(self.w)
        for i in xrange(n):
            V.w[i] = self.w[i]

    def add_from(self, V):
        n = len(self.w)
        for i in xrange(n):
            self.w[i] = V.w[i]

    def add_from_scaled(self, V, a):
        n = len(self.w)
        for i in xrange(n):
            self.w[i] = V.w[i] * a

    def set_const(self, a):
        n = len(self.w)
        for i in xrange(n):
            self.w[i] = a

    def to_json(self):
        # todo: we may want to only save d most significant digits to save space
        # we wont back up gradients to save space
        return {
            'sx': self.sx,
            'sy': self.sy,
            'depth': self.depth,
            'w': self.w,
        }

    def from_json(self, json):
        self.sx = json['sx']
        self.sy = json['sy']
        self.depth = json['depth']

        n = self.sx * self.sy * self.depth
        self.w = [0] * n
        self.dw = [0] * n
        w = json['w']
        # copy over the elements.
        for i in xrange(n):
            self.w[i] = w[i]


class ConvLayer(object):
    def __init__(self, **opt):
        # required
        self.out_dept = opt.get("filters")
        self.sx = opt.get("sx")
        self.in_depth = opt.get("in_depth")
        self.in_sx = opt.get("in_sx")
        self.in_sy = opt.get("in_sy")

        # optional
        self.sy = opt.get("sy", self.sx)
        self.stride = opt.get("stride", 1)
        self.pad = opt.get("pad", 0)
        self.l1_decay_mul = opt.get("l1_decay_mul", 0.0)
        self.l2_decay_mul = opt.get("l2_decay_mul", 1.0)

        # computed
        # note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        # volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        # final application.
        self.out_sx = math.floor((self.in_sx + self.pad * 2 - self.sx) / self.stride + 1)
        self.out_sy = math.floor((self.in_sy + self.pad * 2 - self.sy) / self.stride + 1)
        self.layer_type = "conv"

        # initializations
        bias = opt.get("bias_pref", 0.0)
        self.filters = [Vol(self.sx, self.sy, self.in_depth) for x in xrange(self.out_depth)]
        self.biases = Vol(1, 1, self.out_depth, bias)

    def forward(self, V, is_training):
        self.in_act = V

        A = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)

        for d in xrange(self.out_depth):
            f = self.filters[d]
            x = -self.pad

            ax = 0
            while ax < self.out_sx:
                y = -self.pad

                ay = 0
                while ay < self.out_sy:
                    # convolve centered at this particular location
                    # could be bit more efficient, going for correctness first
                    a = 0.0

                    for fx in xrange(f.sx):
                        for fy in xrange(f.sy):
                            for fd in xrange(f.depth):
                                oy = y + fy  # coordinates in the original input array coordinates
                                ox = x + fx
                                if oy >= 0 and oy < V.sy and ox >= 0 and ox < V.sx:
                                    a += f.w[((f.sx * fy) + fx) * f.depth + fd] * V.w[((V.sx * oy) + ox) * V.depth + fd]

                    a += self.biases.w[d]
                    A.set(ax, ay, d, a)

                    y += self.stride
                    ay += 1

                x += self.stride
                ax += 1

        self.out_act = A
        return A


    def backward(self, y=None):

        # compute gradient wrt weights, biases and input data
        V = self.in_act
        V.dw = [0] * len(V.w)  # zero out gradient wrt bottom data, we're about to fill it
        for d in xrange(self.out_depth):
            f = self.filters[d]
            x = -self.pad

            ax = 0
            while ax < self.out_sx:

                y = -self.pad
                ay = 0

                while ay < self.out_sy:
                    # convolve and add up the gradients. 
                    # could be more efficient, going for correctness first
                    chain_grad = self.out_act.get_grad(ax, ay, d)  # gradient from above, from chain rule

                    for fx in xrange(f.sx):
                        for fy in xrange(f.sy):
                            for fd in xrange(f.depth):
                                oy = y + fy  # coordinates in the original input array coordinates
                                ox = x + fx

                                if oy >= 0 and oy < V.sy and ox >= 0 and ox < V.sx:
                                    # forward prop calculated: a += f.get(fx, fy, fd) * V.get(ox, oy, fd);
                                    # f.add_grad(fx, fy, fd, V.get(ox, oy, fd) * chain_grad);
                                    # V.add_grad(ox, oy, fd, f.get(fx, fy, fd) * chain_grad);

                                    # avoid function call overhead and use Vols directly for efficiency
                                    ix1 = ((V.sx * oy) + ox) * V.depth + fd
                                    ix2 = ((f.sx * fy) + fx) * f.depth + fd

                                    f.dw[ix2] += V.w[ix1] * chain_grad
                                    V.dw[ix1] += f.w[ix2] * chain_grad

                    self.biases.dw[d] += chain_grad

                    y += self.stride
                    ay += 1

                x += self.stride
                ax += 1

    def get_params_and_grads(self):
        response = [
            {'params': f.w, 'grads': f.dw, 'l2_decay_mul': self.l2_decay_mul, 'l1_decay_mul': self.l1_decay_mul}
            for f in self.filters
        ]
        response.append(
            {'params': self.biases.w, 'grads': self.biases.dw, 'l2_decay_mul': 0.0, 'l1_decay_mul': 0.0}
        )
        return response

    def to_json(self):
        return {
            'sx': self.sx,
            'sy': self.sy,
            'stride': self.stride,
            'in_depth': self.in_depth,
            'out_depth': self.out_depth,
            'out_sx': self.out_sx,
            'out_sy': self.out_sy,
            'layer_type': self.layer_type,
            'l1_decay_mul': self.l1_decay_mul,
            'l2_decay_mul': self.l2_decay_mul,
            'pad': self.pad,
            'filters': [f.to_json() for f in self.filters],
            'biases': self.biases.to_json()
        }

    def from_json(self, json):
        self.out_depth = json['out_depth']
        self.out_sx = json['out_sx']
        self.out_sy = json['out_sy']
        self.layer_type = json['layer_type']
        self.sx = json['sx']
        self.sy = json['sy']
        self.stride = json['stride']
        self.in_depth = json['in_depth']
        self.l1_decay_mul = json.get('l1_decay_mul', 1.0)
        self.l2_decay_mul = json.get('l2_decay_mul', 1.0)
        self.pad = json.get('pad', 0)
        self.filters = []
        for f in json['filters']:
            v = Vol(0, 0, 0, 0)
            v.from_json(f)
            self.filters.append(v)
        self.biases = Vol(0, 0, 0, 0)
        self.biases.from_json(json['biases'])


class FullyConnLayer(object):
    def __init__(self, **opt):

        # required
        # ok fine we will allow 'filters' as the word as well
        self.out_depth = opt.get("num_neurons", opt.get("filters"))

        # optional
        self.l1_decay_mul = opt.get("l1_decay_mul", 0.0)
        self.l2_decay_mul = opt.get("l2_decay_mul", 1.0)

        # computed
        self.num_inputs = opt.get("in_sx") * opt.get("in_sy") * opt.get("in_depth")
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = "fc"

        # initializations
        bias = opt.get("bias_pref", 0.0)
        self.filters = [Vol(1, 1, self.num_inputs) for x in xrange(self.out_depth)]
        self.biases = Vol(1, 1, self.out_depth, bias)

    def forward(self, V, is_training):
        self.in_act = V
        A = Vol(1, 1, self.out_depth, 0.0)
        Vw = V.w
        for i in xrange(self.out_depth):
            a = 0.0
            wi = self.filters[i].w
            for d in xrange(self.num_inputs):
                a += Vw[d] * wi[d]  # for efficiency use Vols directly for now
            a += self.biases.w[i]
            A.w[i] = a
        self.out_act = A
        return A

    def backward(self, y=None):
        V = self.in_act
        V.dw = [0] * len(V.w)  # zero out the gradient in input Vol
        # compute gradient wrt weights and data
        for i in xrange(self.out_depth):
            tfi = self.filters[i]
            chain_grad = self.out_act.dw[i]
            for d in xrange(self.num_inputs):
                V.dw[d] += tfi.w[d] * chain_grad  # grad wrt input data
                tfi.dw[d] += V.w[d] * chain_grad  # grad wrt params
            self.biases.dw[i] += chain_grad

    def get_params_and_grads(self):
        response = [
            {'params': f.w, 'grads': f.dw, 'l2_decay_mul': self.l2_decay_mul, 'l1_decay_mul': self.l1_decay_mul}
            for f in self.filters
        ]
        response.append(
            {'params': self.biases.w, 'grads': self.biases.dw, 'l2_decay_mul': 0.0, 'l1_decay_mul': 0.0}
        )
        return response

    def to_json(self):
        return {
            'out_depth': self.out_depth,
            'out_sx': self.out_sx,
            'out_sy': self.out_sy,
            'layer_type': self.layer_type,
            'num_inputs': self.num_inputs,
            'l1_decay_mul': self.l1_decay_mul,
            'l2_decay_mul': self.l2_decay_mul,
            'filters': [f.to_json() for f in self.filters],
            'biases': self.biases.to_json()
        }

    def from_json(self, json):
        self.out_depth = json['out_depth']
        self.out_sx = json['out_sx']
        self.out_sy = json['out_sy']
        self.layer_type = json['layer_type']
        self.num_inputs = json['num_inputs']
        self.l1_decay_mul = json.get('l1_decay_mul', 1.0)
        self.l2_decay_mul = json.get('l2_decay_mul', 1.0)
        self.pad = json.get('pad', 0)
        self.filters = []
        for f in json['filters']:
            v = Vol(0, 0, 0, 0)
            v.from_json(f)
            self.filters.append(v)
        self.biases = Vol(0, 0, 0, 0)
        self.biases.from_json(json['biases'])


# Class PoolLayer # line 539 convnet.js

class InputLayer(object):
    def __init__(self, **opts):
        self.out_sx = opts.get("out_sx", opts.get("in_sx"))
        self.out_sy = opts.get("out_sy", opts.get("in_sy"))
        self.out_depth = opts.get("out_depth", opts.get("in_depth"))
        self.layer_type = "input"

    def forward(self, V, is_training):
        self.in_act = V
        self.out_act = V
        return V

    def backward(self, y=None):
        pass

    def get_params_and_grads(self):
        return []

    def to_json(self):
        return {
            'out_depth': self.out_depth,
            'out_sx': self.out_sx,
            'out_sy': self.out_sy,
            'layer_type': self.layer_type,
        }

    def from_json(self, json):
        self.out_depth = json['out_depth']
        self.out_sx = json['out_sx']
        self.out_sy = json['out_sy']
        self.layer_type = json['layer_type']


class SoftmaxLayer(object):
    def __init__(self, **opts):

        self.num_inputs = opts.get("in_sx") * opts.get("in_sy") * opts.get("in_depth")
        self.out_depth = self.num_inputs
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = "softmax"

    def forward(self, V, is_training):
        self.in_act = V

        A = Vol(1, 1, self.out_depth, 0.0)

        # compute max activation
        amax = max(V.w)

        # compute exponentials (carefully to not blow up)
        es = [0] * self.out_depth
        esum = 0.0
        for i in xrange(self.out_depth):
            e = math.exp(V.w[i] - amax)
            esum += e
            es[i] = e

        # normalize and output to sum to one
        for i in xrange(self.out_depth):
            es[i] /= esum
            A.w[i] = es[i]

        self.es = es  # save these for backprop
        self.out_act = A
        return A

    def backward(self, y=None):

        # compute and accumulate gradient wrt weights and bias of this layer
        x = self.in_act
        x.dw = [0] * len(x.w)  # zero out the gradient of input Vol

        for i in xrange(self.out_depth):
            indicator = 1.0 if i == y else 0.0
            mul = -(indicator - self.es[i])
            x.dw[i] = mul

        # loss is the class negative log likelihood
        return -math.log(self.es[y])

    def get_params_and_grads(self):
        return []

    def to_json(self):
        return {
            'out_depth': self.out_depth,
            'out_sx': self.out_sx,
            'out_sy': self.out_sy,
            'layer_type': self.layer_type,
            'num_inputs': self.num_inputs,
        }

    def from_json(self, json):
        self.out_depth = json['out_depth']
        self.out_sx = json['out_sx']
        self.out_sy = json['out_sy']
        self.layer_type = json['layer_type']
        self.num_inputs = json['num_inputs']


# class RegressionLayer # line 798 convnet.js
# class SVMLayer # line 860 convnet.js
# class ReluLayer # line 935 convnet.js
# class SigmoidLayer # line 988 convnet.js
# class MaxoutLayer # line 1043 convnet.js


class TanhLayer(object):
    def __init__(self, **opts):

        self.out_depth = opts.get("in_depth")
        self.out_sx = opts.get("in_sx")
        self.out_sy = opts.get("in_sy")
        self.layer_type = "tanh"

    def forward(self, V, is_training):
        self.in_act = V
        V2 = V.clone_and_zero()
        for i in xrange(len(V.w)):
            V2.w[i] = math.tanh(V.w[i])
        self.out_act = V2
        return V2

    def backward(self, y=None):
        V = self.in_act  # we need to set dw of this
        V2 = self.out_act
        n = len(V.w)
        V.dw = [0] * n
        for i in xrange(n):
            v2wi = V2.w[i]
            V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i]

    def get_params_and_grads(self):
        return []

    def to_json(self):
        return {
            'out_depth': self.out_depth,
            'out_sx': self.out_sx,
            'out_sy': self.out_sy,
            'layer_type': self.layer_type,
        }

    def from_json(self, json):
        self.out_depth = json['out_depth']
        self.out_sx = json['out_sx']
        self.out_sy = json['out_sy']
        self.layer_type = json['layer_type']


# class DropoutLayer # line 1229 convnet.js
# class LocalResponseNormalizationLayer # line 1301 convnet.js
# class QuadTransformLayer # line 1414 convnet.js


LAYERS = {
    "fc": FullyConnLayer,
    "lrn": None,
    "dropout": None,
    "input": InputLayer,
    "softmax": SoftmaxLayer,
    "regression": None,
    "conv": ConvLayer,
    "pool": None,
    "relu": None,
    "sigmoid": None,
    "tanh": TanhLayer,
    "maxout": None,
    "quadtransform": None,
    "svm": None,
}


class Net(object):
    def __init__(self, **opts):
        self.layers = []

    def make_layers(self, defs):

        if len(defs) < 2:
            raise ValueError("ERROR! For now at least have input and softmax layers.")
        if defs[0]["type"] != "input":
            raise ValueError("ERROR! For now first layer should be input.")

        def desugar(defs):
            new_defs = []
            for d in defs:
                dtype = d["type"]
                if dtype == "softmax" or dtype == "svm":
                    # add an fc layer here, there is no reason the user should
                    # have to worry about this and we almost always want to
                    new_defs.append({"type": "fc", "num_neurons": d["num_classes"]})

                if dtype == "regression":
                    # add an fc layer here, there is no reason the user should
                    # have to worry about this and we almost always want to
                    new_defs.append({"type": "fc", "num_neurons": d["num_neurons"]})

                if (dtype == "fc" or dtype == "conv") and d.get("bias_pref") is None:
                    if d.get("activation") == "relu":
                        # relus like a bit of positive bias to get gradients early
                        # otherwise it's technically possible that a relu unit will never turn on (by chance)
                        # and will never get any gradient and never contribute any computation. Dead relu.
                        d["bias_pref"] = 0.1
                    else:
                        d["bias_pref"] = 0.0

                if d.get("tensor") is not None:
                    # apply quadratic transform so that the upcoming multiply will include
                    # quadratic terms, equivalent to doing a tensor product
                    tensor = d.get("tensor")
                    if tensor:
                        new_defs.append({"type": "quadtransform"})

                new_defs.append(d)

                activation = d.get("activation")
                if activation is not None:
                    if activation == "relu":
                        new_defs.append({"type": "relu"})
                    elif activation == "sigmoid":
                        new_defs.append({"type": "sigmoid"})
                    elif activation == "tanh":
                        new_defs.append({"type": "tanh"})
                    elif activation == "maxout":
                        # create maxout activation, and pass along group size, if provided
                        gs = d.get("group_size", 2)
                        new_defs.append({"type": "maxout", "group_size": gs})
                    else:
                        raise ValueError("ERROR unsupported activation %s" % (activation))

                if d.get("drop_prob") is not None and dtype == "dropout":
                    new_defs.append({"type": "dropout", "drop_prob": d.get("drop_prob")})

            return new_defs

        defs = desugar(defs)

        # create the layers
        self.layers = []
        for index, d in enumerate(defs):
            if index > 0:
                prev = self.layers[index - 1]
                d["in_sx"] = prev.out_sx
                d["in_sy"] = prev.out_sy
                d["in_depth"] = prev.out_depth

            dtype = d.get("type")

            if dtype in LAYERS:
                klass = LAYERS.get(dtype)
                if klass is None:
                    raise NotImplementedError("ERROR Layer type not implemented %s" % (dtype))
                self.layers.append(klass(**d))
            else:
                raise ValueError("ERROR: UNRECOGNIZED LAYER TYPE! %s" % (dtype))

    def forward(self, V, is_training=False):
        """ forward prop the network. A trainer will pass in is_training = true """
        act = self.layers[0].forward(V, is_training)

        for layer in self.layers[1:]:
            act = layer.forward(act, is_training)

        return act

    def backward(self, y):
        loss = self.layers[-1].backward(y=y)
        for layer in self.layers[-2::-1]:
            layer.backward()
        return loss

    def get_params_and_grads(self):
        response = []
        for layer in self.layers:
            layer_response = layer.get_params_and_grads()
            response.extend(layer_response)
        return response


    def get_prediction(self):
        S = self.layers[-1]  # softmax layer
        p = S.out_act.w
        maxv = p[0]
        maxi = 0
        for index, pi in p[1:]:
            if pi > maxv:
                maxi, maxv = index, pi

        return maxi

    def to_json(self):
        return {
            "layers": [layer.to_json() for layer in self.layers]
        }

    def from_json(self, json):
        self.layers = []
        for Lj in json["layers"]:
            t = Lj["layer_type"]
            L = LAYERS.get(t)()
            L.from_json(Lj)
            self.layers.append(L)


class Trainer(object):
    def __init__(self, net, **opts):

        self.net = net

        self.learning_rate = opts.get("learning_rate", 0.01)
        self.l1_decay = opts.get("l1_decay", 0.0)
        self.l2_decay = opts.get("l2_decay", 0.0)
        self.batch_size = opts.get("batch_size", 1)
        self.method = opts.get("method", "sgd")  # sgd/adagrad/adadelta/windowgrad

        self.momentum = opts.get("momentum", 0.01)
        self.ro = opts.get("ro", 0.01)  # used in adadelta
        self.eps = opts.get("eps", 0.01)  # used in adadelta

        self.k = 0  # iteration counter
        self.gsum = []  # last iteration gradients (used for momentum calculations)
        self.xsum = []  # used in adadelta

    def train(self, x, y):

        start = time.time()
        self.net.forward(x, True)  # also set the flag that lets the net know we're just training
        end = time.time()

        fwd_time = end - start

        start = time.time()
        cost_loss = self.net.backward(y)
        l2_decay_loss = 0.0
        l1_decay_loss = 0.0
        end = time.time()

        bwd_time = end - start

        self.k += 1

        if not self.k % self.batch_size:

            pglist = self.net.get_params_and_grads()

            # initialize lists for accumulators. Will only be done once on first iteration
            if not self.gsum and (self.method != 'sgd' or self.momentum > 0.0):
                # only vanilla sgd doesnt need either lists
                # momentum needs gsum
                # adagrad needs gsum
                # adadelta needs gsum and xsum
                for pg in pglist:
                    params = pg["params"]
                    N = len(params)
                    self.gsum.append([0] * N)
                    if self.method == "adadelta":
                        self.xsum.append([0] * N)
                    else:
                        self.xsum.append([])  # conserve memory

            # perform an update for all sets of weights
            # param, gradient, other options in future (custom learning rate etc)
            for i, pg in enumerate(pglist):
                p = pg["params"]
                g = pg["grads"]

                # learning rate for some parameters.
                l2_decay_mul = pg.get("l2_decay_mul", 1.0)
                l1_decay_mul = pg.get("l1_decay_mul", 1.0)
                l2_decay = l2_decay_mul * self.l2_decay
                l1_decay = l1_decay_mul * self.l1_decay

                for j, pj in enumerate(p):
                    l2_decay_loss += l2_decay * pj * pj / 2  # accumulate weight decay loss
                    l1_decay_loss += l1_decay * abs(pj)
                    l1grad = l1_decay * (1 if pj > 0 else -1)
                    l2grad = l2_decay * (pj)

                    gij = (l2grad + l1grad + g[j]) / self.batch_size  # raw batch gradient

                    gsumi = self.gsum[i]
                    xsumi = self.xsum[i]
                    if self.method == "adagrad":
                        gsumi[j] = gsumi[j] + gij * gij
                        dx = -self.learning_rate / math.sqrt(gsumi[j] + self.eps) * gij
                        p[j] += dx  # affect p[j], not pj
                    elif self.method == "windowgrad":
                        # this is adagrad but with a moving window weighted average
                        # so the gradient is not accumulated over the entire history of the run. 
                        # it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                        gsumi[j] = self.ro * gsumi[j] + (1 - self.ro) * gij * gij
                        dx = -self.learning_rate / math.sqrt(
                            gsumi[j] + self.eps) * gij  # eps added for better conditioning
                        p[j] += dx  # affect p[j], not pj
                    elif self.method == "adadelta":
                        # assume adadelta if not sgd or adagrad
                        gsumi[j] = self.ro * gsumi[j] + (1 - self.ro) * gij * gij
                        dx = -math.sqrt((xsumi[j] + self.eps) / (gsumi[j] + self.eps)) * gij
                        xsumi[j] = self.ro * xsumi[j] + (1 - self.ro) * dx * dx  # yes, xsum lags behind gsum by 1.
                        p[j] += dx  # affect p[j], not pj
                    else:
                        # assume SGD
                        if self.momentum > 0.0:
                            dx = self.momentum * gsumi[j] - self.learning_rate * gij  # step
                            gsumi[j] = dx  # back this up for next iteration of momentum
                            p[j] += dx  # apply correted gradient
                        else:
                            p[j] += -self.learning_rate * gij

                    g[j] = 0.0  # zero out gradient so that we can begin accumulating anew

        # appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
        # in future, TODO: have to completely redo the way loss is done around the network as currently 
        # loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
        # and it should all be computed correctly and automatically. 
        return {
            "fwd_time": fwd_time,
            "bwd_time": bwd_time,
            "l2_decay_loss": l2_decay_loss,
            "l1_decay_loss": l1_decay_loss,
            "cost_loss": cost_loss,
            "softmax_loss": cost_loss,
            "loss": cost_loss + l1_decay_loss + l2_decay_loss,
        }


SGDTrainer = Trainer  # backwards compatiblity





















