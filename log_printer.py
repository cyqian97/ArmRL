import numpy as np

# Replace 'your_log_file.npz' with the actual path to your file
with np.load('results/ppo_oobs_Lift/log/evaluations.npz') as data:
    # 'data' now behaves like a dictionary, with keys representing the names of the arrays
    # stored within the .npz file.
    
    # To see the names of the arrays (keys) in the .npz file:
    print("Keys in the .npz file:", data.files)

    # To access a specific array, use its key:
    # For example, if you found a key named 'arr_0' from data.files:
    # log_array = data['arr_0']
    # print("Contents of 'arr_0':", log_array)

    # You can iterate through all arrays and print their contents:
    for key in data.keys():
        print(f"\nContents of array '{key}':")
        print(data[key])