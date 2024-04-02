import numpy as np

def HGluv2xyz(luv, white):
    if luv.shape[1] != 3:
        print('luv must be n by 3')
        return None

    xyz = np.zeros_like(luv)

    # compute u' v' for white
    upw = 4 * white[0] / (white[0] + 15 * white[1] + 3 * white[2])
    vpw = 9 * white[1] / (white[0] + 15 * white[1] + 3 * white[2])

    # compute Y
    index = (luv[:, 0] > 8)
    xyz[:, 1] = xyz[:, 1] + index * (((luv[:, 0] + 16) / 116) ** 3)
    xyz[:, 1] = xyz[:, 1] + (1 - index) * (luv[:, 0] / 903.3)

    # compute a, b, c, d
    a = (1 / 3) * (52 * luv[:, 0] / (luv[:, 1] + 13 * luv[:, 0] * upw) - 1)
    b = -5 * xyz[:, 1]
    c = -1 / 3
    d = xyz[:, 1] * (39 * luv[:, 0] / (luv[:, 2] + 13 * luv[:, 0] * vpw) - 5)

    # compute X,Z
    xyz[:, 0] = (d - b) / (a - c)
    xyz[:, 2] = xyz[:, 0] * a + b

    return xyz

# Example usage:
luv_values = np.array([[50, 0, 0], [70, 10, -20]])  # Example LUV values
white_point = [0.95047, 1.0, 1.08883]  # Example white point in XYZ
xyz_values = HGluv2xyz(luv_values, white_point)
print(xyz_values)
