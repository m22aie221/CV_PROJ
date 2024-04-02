import numpy as np

def lab2xyz(lab, obs, xyzw):
    def _compute_xyz(lab, white):
        xyz = np.zeros_like(lab)
        
        # Compute Y
        index = (lab[:,0] > 7.9996)
        xyz[:,1] += index * (white[1] * ((lab[:,0] + 16) / 116) ** 3)
        xyz[:,1] += (1 - index) * (white[1] * lab[:,0] / 903.3)
        
        # Compute fy for use later
        fy = xyz[:,1] / white[1]
        index = (fy > 0.008856)
        fy = np.zeros_like(lab[:,0])
        fy += index * (xyz[:,1] / white[1]) ** (1 / 3)
        fy += (1 - index) * (7.787 * xyz[:,1] / white[1] + 16 / 116)
        
        # Compute X
        index = ((lab[:,1] / 500 + fy) ** 3 > 0.008856)
        xyz[:,0] += index * (white[0] * (lab[:,1] / 500 + fy) ** 3)
        xyz[:,0] += (1 - index) * (white[0] * ((lab[:,1] / 500 + fy) - 16 / 116) / 7.787)
        
        # Compute Z
        index = ((fy - lab[:,2] / 200) ** 3 > 0.008856)
        xyz[:,2] += index * (white[2] * (fy - lab[:,2] / 200) ** 3)
        xyz[:,2] += (1 - index) * (white[2] * ((fy - lab[:,2] / 200) - 16 / 116) / 7.787)
        
        return xyz
    
    if obs == 'user':
        white = xyzw
    elif obs in ['a_64', 'a_31', 'c_64', 'c_31', 'd50_64', 'd50_31', 'd55_64', 'd55_31', 'd65_64', 'd65_31', 'd75_64', 'd75_31', 'f2_64', 'f2_31', 'f7_64', 'f7_31', 'f11_64', 'f11_31']:
        if obs == 'a_64':
            white = [111.144, 100.00, 35.200]
        elif obs == 'a_31':
            white = [109.850, 100.00, 35.585]
        elif obs == 'c_64':
            white = [97.285, 100.00, 116.145]
        elif obs == 'c_31':
            white = [98.074, 100.00, 118.232]
        elif obs == 'd50_64':
            white = [96.720, 100.00, 81.427]
        elif obs == 'd50_31':
            white = [96.422, 100.00, 82.521]
        elif obs == 'd55_64':
            white = [95.799, 100.00, 90.926]
        elif obs == 'd55_31':
            white = [95.682, 100.00, 92.149]
        elif obs == 'd65_64':
            white = [94.811, 100.00, 107.304]
        elif obs == 'd65_31':
            white = [95.047, 100.00, 108.883]
        elif obs == 'd75_64':
            white = [94.416, 100.00, 120.641]
        elif obs == 'd75_31':
            white = [94.072, 100.00, 122.638]
        elif obs == 'f2_64':
            white = [103.279, 100.00, 69.027]
        elif obs == 'f2_31':
            white = [99.186, 100.00, 67.393]
        elif obs == 'f7_64':
            white = [95.792, 100.00, 107.686]
        elif obs == 'f7_31':
            white = [95.041, 100.00, 108.747]
        elif obs == 'f11_64':
            white = [103.863, 100.00, 65.607]
        elif obs == 'f11_31':
            white = [100.962, 100.00, 64.350]
    else:
        print('Unknown option obs')
        print('Use d65_64 for D65 and 1964 observer')
        return None
    
    if lab.shape[1] != 3:
        print('lab must be n by 3')
        return None
    
    xyz = _compute_xyz(lab, white)
    return xyz

# Example usage:
'''lab_values = np.array([[50, 0, 0], [70, 10, -20]])  # Example LAB values
observer = 'd65_64'  # Example observer
white_point = [94.811, 100.00, 107.304]  # Example white point for D65 1964
xyz_values = lab2xyz(lab_values, observer, white_point)
print(xyz_values)'''
