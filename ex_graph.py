#!/usr/bin/env python
import math
import sys
import time

import pygame

from convnet import convnet

WIDTH = 800
HALF_WIDTH = WIDTH / 2
HEIGHT = 400
ITERS = 2
N = 10

BG_COLOR = (200, 200, 200)
GREEN = (150, 250, 150)
RED = (250, 150, 150)
GRAY = (50, 50, 50)
BLACK = (0, 0, 0)
GREEN_PT = (100, 200, 100)
RED_PT = (200, 100, 100)


def random_data(minv=-3, maxv=3):
    global N
    data = [(convnet.randf(minv, maxv), convnet.randf(minv, maxv)) for x in xrange(N)]
    labels = [1 if convnet.randf(0, 1) > 0.5 else 0 for i in xrange(N)]  # Randomly assign red / green
    return data, labels


def spiral_data(minv=-0.1, maxv=0.1):
    global N
    data = []
    labels = []

    for i in xrange(N / 2):
        r = i / N * 5 + convnet.randf(minv, maxv)
        t = 1.25 * i / N * 2 * math.pi + convnet.randf(minv, maxv)
        data.append((r * math.sin(t), r * math.cos(t)))
        labels.append(1)

    for i in xrange(N / 2):
        r = i / N * 5 + convnet.randf(minv, maxv)
        t = 1.25 * i / N * 2 * math.pi + math.pi + convnet.randf(minv, maxv)
        data.append((r * math.sin(t), r * math.cos(t)))
        labels.append(0)

    return data, labels


data, labels = random_data()


def on_iteration(window, net, trainer, speed):
    window.fill(BG_COLOR)
    draw_frame(window, net, trainer)
    pygame.display.flip()


ss = 50.0
lix = 4  #  layer id to track first 2 neurons of
d0 = 0  #  first dimension to show visualized
d1 = 1  #  second dimension to show visualized
density = 5
gridstep = 2
sz = density * gridstep


def draw_frame(window, net, trainer):
    global lix, d0, d1, ss, sz, gridstep, density, data, labels

    netx = convnet.Vol(1, 1, 2)
    gridx = []
    gridy = []
    gridl = []


    # draw decisions in the grid
    x = 0.0
    cx = 0
    while x <= HALF_WIDTH:

        y = 0.0
        cy = 0
        while y <= HEIGHT:

            netx.w[0] = (x - HALF_WIDTH / 2) / ss
            netx.w[1] = (y - HEIGHT / 2) / ss

            a = net.forward(netx, False)

            color = RED if a.w[0] > a.w[1] else GREEN

            pygame.draw.rect(window, color, (x - density / 2 - 1, y - density / 2 - 1, density + 2, density + 2))

            if cx % gridstep == 0 and cy % gridstep == 0:
                xt = net.layers[lix].out_act.w[d0]  # in screen coords
                yt = net.layers[lix].out_act.w[d1]  # in screen coords
                gridx.append(xt)
                gridy.append(yt)
                gridl.append(a.w[0] > a.w[1])  # remember final label as well

            y += density
            cy += 1

        x += density
        cx += 1

    # draw axes
    pygame.draw.line(window, GRAY, (0, HEIGHT / 2), (HALF_WIDTH, HEIGHT / 2))
    pygame.draw.line(window, GRAY, (HALF_WIDTH / 2, 0), (HALF_WIDTH / 2, HEIGHT))

    # draw representation transformation axes for two neurons at some layer

    mmx = convnet.maxim(gridx)
    mmy = convnet.maxim(gridy)
    ng = len(gridx)
    n = int(math.floor(math.sqrt(ng)))
    for x in xrange(n):
        for y in xrange(n):
            # down
            ix1 = x * n + y
            ix2 = x * n + y + 1
            if ix1 >= 0 and ix2 >= 0 and ix1 < ng and ix2 < ng and y < n - 1:  # check oob
                xraw1 = HALF_WIDTH + HALF_WIDTH * (gridx[ix1] - mmx.minv) / mmx.dv
                xraw2 = HALF_WIDTH + HALF_WIDTH * (gridx[ix2] - mmx.minv) / mmx.dv
                yraw1 = HEIGHT * (gridy[ix1] - mmy.minv) / mmy.dv
                yraw2 = HEIGHT * (gridy[ix2] - mmy.minv) / mmy.dv
                pygame.draw.line(window, BLACK, (xraw1, yraw1), (xraw2, yraw2))

            # and draw its color
            color = GREEN if gridl[ix1] else RED
            pygame.draw.rect(window, color, (xraw1 - sz / 2 - 1, yraw1 - sz / 2 - 1, sz + 2, sz + 2))

            # right
            ix1 = (x + 1 * n + y)
            ix2 = x * n + y
            if ix1 >= 0 and ix2 >= 0 and ix1 < ng and ix2 < ng and x < n - 1:  # check oob
                xraw1 = HALF_WIDTH + HALF_WIDTH * (gridx[ix1] - mmx.minv) / mmx.dv
                xraw2 = HALF_WIDTH + HALF_WIDTH * (gridx[ix2] - mmx.minv) / mmx.dv
                yraw1 = HEIGHT * (gridy[ix1] - mmy.minv) / mmy.dv
                yraw2 = HEIGHT * (gridy[ix2] - mmy.minv) / mmy.dv
                pygame.draw.line(window, BLACK, (xraw1, yraw1), (xraw2, yraw2))

    # draw datapoints
    for i in xrange(N):
        color = GREEN_PT if labels[i] == 1 else RED_PT
        pygame.draw.circle(window, color, map(int, (data[i][0] * ss + HALF_WIDTH / 2, data[i][1] * ss + HEIGHT / 2)), 5)

        # # also draw transformed data points while we're at it
        # netx.w[0] = data[i][0]
        # netx.w[1] = data[i][1]
        #
        # a = net.forward(netx, False)
        # xt = HALF_WIDTH + HALF_WIDTH * (net.layers[lix].out_act.w[d0] - mmx.minv) / mmx.dv  # in screen coords
        # yt = HEIGHT * (net.layers[lix].out_act.w[d1] - mmy.minv) / mmy.dv  # in screen coords
        # pygame.draw.circle(window, color, map(int, (xt, yt)), 5)


def main():
    global data, labels

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.time.set_timer(pygame.USEREVENT, 1)  # Re-fire ASAP

    ## NET

    net = convnet.Net()
    net.make_layers([
        {"type": 'input', "out_sx": 1, "out_sy": 1, "out_depth": 2},
        {"type": 'fc', "num_neurons": 6, "activation": 'tanh'},
        {"type": 'fc', "num_neurons": 2, "activation": 'tanh'},
        {"type": 'softmax', "num_classes": 2},
    ])

    trainer = convnet.Trainer(net, learning_rate=0.01, momentum=0.1, batch_size=10, l2_decay=0.001)

    k = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.USEREVENT:

                start = time.time()
                x = convnet.Vol(1, 1, 2)
                for iters in xrange(ITERS):
                    for ix in xrange(N):
                        x.w = data[ix]
                        trainer.train(x, labels[ix])
                end = time.time()

                k += 1

                if k % 10 == 0:
                    on_iteration(window, net, trainer, end - start)


if __name__ == "__main__":
    main()