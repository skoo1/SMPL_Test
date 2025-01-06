import numpy as np
from scipy.io import loadmat

def mat_to_npz(mat_file, npz_file):
    """
    Convert a MATLAB .mat file to a NumPy .npz file.
    """
    # 1. Load the .mat data
    mat_data = loadmat(mat_file)
    
    # 2. Create a dictionary to hold only the variables we need
    #    (we'll ignore __header__, __version__, and __globals__ which are default mat keys)
    variables_to_save = {}
    
    # List the variable names you want from the .mat file
    var_names = [
        'J_regressor',
        'f',
        'kintree_table',
        'posedirs',
        'shapedirs',
        'v_template',
        'weights'
    ]
    
    for var_name in var_names:
        if var_name in mat_data:
            variables_to_save[var_name] = mat_data[var_name]
        else:
            print(f"Warning: {var_name} not found in {mat_file}")
    
    # 3. Save the variables to an .npz file
    np.savez(npz_file, **variables_to_save)

if __name__ == "__main__":
    # Example usage:
    mat_file_path = 'model.mat'   # path to your .mat file
    npz_file_path = 'model.npz'   # desired .npz output file path

    mat_to_npz(mat_file_path, npz_file_path)
    print(f"Saved {mat_file_path} to {npz_file_path}")
