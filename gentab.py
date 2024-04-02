"""
This Python implementation uses NumPy and Pandas libraries. It defines a summary function to compute summary statistics along the specified axis. Then, it computes the mean, median, 95th percentile, and maximum values of the errors for each method. Finally, it creates a Pandas DataFrame to store the statistics and displays the table if de_str is provided.
"""
import numpy as np
import pandas as pd

def gentab(de, Method, de_str=None):
    # Check if de_str is provided to determine if the table should be displayed
    disp_flag = True if de_str is not None else False
    
    # Check dimensions of Method and de
    if len(Method) != de.shape[2]:
        raise ValueError("Dimension of methods mismatches the error matrix.")
    
    # Define summary function
    def summary(f, x):
        return np.squeeze(np.mean(f(x, axis=1), axis=2))
    
    # Compute statistics
    de_mean = summary(np.mean, de)
    de_median = summary(np.median, de)
    de_95 = summary(lambda x: np.percentile(x, 95, axis=1), de)
    de_max = summary(np.amax, de)
    
    # Create DataFrame
    Variable = ['Mean', 'Median', 'pct95', 'Max']
    tab = pd.DataFrame({'Mean': de_mean, 'Median': de_median,
                        'pct95': de_95, 'Max': de_max}, 
                       index=Method)
    
    # Display table if disp_flag is True
    if disp_flag:
        print(de_str)
        print(tab)
    
    return tab
