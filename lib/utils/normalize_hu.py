import numpy as np
def cut_and_normalize_hu(cube, window_width=1500, window_level=-400, cube_reshape=[32, 48, 48],augment_prob=0.0):
    d, h, w, *_ = cube.shape
    center = np.array([d / 2, h / 2, w / 2], dtype=np.int)
    trans = np.random.randint(-5, 5, 3)
    if augment_prob > 0:
        center += trans
    cube = cube[center[0] - int(0.5 * cube_reshape[0]):center[0] + int(0.5 * cube_reshape[0]),
            center[1] - int(0.5 * cube_reshape[1]):center[1] + int(0.5 * cube_reshape[1]),
            center[2] - int(0.5 * cube_reshape[2]):center[2] + int(0.5 * cube_reshape[2])]
    # here we need to use the original HU value
    value_min = window_level - window_width / 2
    value_max = window_level + window_width / 2
    cube = (cube - value_min) / (value_max - value_min)
    cube[cube > 1] = 1
    cube[cube < 0] = 0
    return cube

def normalize_hu(cube, window_width=1500, window_level=-400):
    value_min = window_level - window_width / 2
    value_max = window_level + window_width / 2
    cube = (cube - value_min) / (value_max - value_min)
    cube[cube > 1] = 1
    cube[cube < 0] = 0
    return cube
