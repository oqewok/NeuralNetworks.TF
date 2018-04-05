import numpy as np


def generate_random_image(shape):
    return np.random.randint(0, 255, size=shape, dtype=int)


if __name__ == '__main__':
    img = generate_random_image([64, 128, 3])

    pass