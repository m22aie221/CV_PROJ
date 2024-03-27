import numpy as np

def luv2rgb(luv):
    # Convert to XYZ
    xyz = luv.copy()

    Yn = 1
    Un_prime = 0.19784977571475
    Vn_prime = 0.46834507665248

    u_prime = luv[1] / (13 * luv[0]) + Un_prime
    v_prime = luv[2] / (13 * luv[0]) + Vn_prime

    xyz[1] = np.where(luv[0] < 8.0, Yn * luv[0] / 903.3, Yn * (((luv[0] + 16.0) / 116.0) ** 3))
    xyz[0] = 9 * u_prime * xyz[1] / (4 * v_prime)
    xyz[2] = (12 - 3 * u_prime - 20 * v_prime) * xyz[1] / (4 * v_prime)

    # Convert to RGB
    RGB = np.array([[3.2405, -1.5371, -0.4985],
                    [-0.9693, 1.8760, 0.0416],
                    [0.0556, -0.2040, 1.0573]])

    rgb = RGB.dot(xyz)

    # Bounds check
    rgb[0] = np.where(luv[0] >= 0.1, rgb[0], 0)
    rgb[1] = np.where(luv[0] >= 0.1, rgb[1], 0)
    rgb[2] = np.where(luv[0] >= 0.1, rgb[2], 0)

    rgb[0] = np.where((rgb[0] >= 0) & (rgb[0] <= 1), rgb[0], rgb[0] > 1)
    rgb[1] = np.where((rgb[1] >= 0) & (rgb[1] <= 1), rgb[1], rgb[1] > 1)
    rgb[2] = np.where((rgb[2] >= 0) & (rgb[2] <= 1), rgb[2], rgb[2] > 1)

    return rgb

'''# Example usage:
luv_values = np.array([50, 0, 0])  # Example LUV values
rgb_values = luv2rgb(luv_values)
print(rgb_values)'''
