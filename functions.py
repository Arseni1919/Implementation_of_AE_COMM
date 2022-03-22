from GLOBALS import *


def distance_nodes(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def distance_points(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def copy_nn_parameters(target_nn, copied_nn):
    # UPDATE OLD NET
    for target_param, param in zip(target_nn.parameters(), copied_nn.parameters()):
        target_param.data.copy_(param.data)
