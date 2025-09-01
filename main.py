from src.models import model as mainmodel
from src import train
from src import loader
import torch
import numpy as np
import argparse
import csv
import os
import matplotlib.pyplot as plt

video_number = 0

def evaluate_model(model, data_train, dt, name):
    dataloader  = loader.getLoader_folder(data_train, split=False)
    z = None

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    model.to(device)

    z = None
    X = []

    for data in dataloader:

        input_Data, out_Data = data

        x0 = input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        x2 = out_Data.to(device=device, dtype=torch.float)

        outputs = model(x0)
        z2_encoder, z2_phys, z3=outputs

        if z is None:
            z = z2_encoder.detach().cpu().numpy()[0][0]
        else:
            z = np.vstack((z,z2_encoder.detach().cpu().numpy()[0][0]))

    
    for i in range(1, z2_encoder.shape[1]):
        z = np.vstack((z,z2_encoder.detach().cpu().numpy()[0][i]))

    alpha = model.pModel.alpha[0].detach().cpu().numpy().item()
    beta = model.pModel.beta[0].detach().cpu().numpy().item()

    print("z2_encoder.shape: ", z2_encoder.shape)
    dt = 1/60
    plt.figure(figsize=(20,5))
    time = np.arange(z.shape[0])
    plt.plot(time*dt, z, label='real', marker='o', linestyle='-')
    #show alpha and beta in text
    plt.text(0.5, 0.5, 'alpha: '+str(alpha)+'\nbeta: '+str(beta), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
 

    plt.xlabel('Time')
    plt.ylabel('z')
    plt.savefig(f'./Results/{name}.png', dpi=300)
    plt.show()
    plt.close()

    # plt.figure(figsize=(20,5))
    # time = np.arange(z.shape[0])
    # time = time*dt
    # h = 1/z

    # dr = 0.5*time*time*alpha+h[0]
    # plt.plot(time, h, label='real', marker='o', linestyle='-')
    # plt.plot(time, dr, label='dr', marker='o', linestyle='-')
    # #show alpha and beta in text
    # plt.text(0.5, 0.5, 'alpha: '+str(alpha)+'\nmaxtime: '+str(max_time*dt), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    # plt.legend()

    # plt.xlabel('Time')
    # plt.ylabel('z')
    # plt.savefig(f'./Results/{name}_h.png', dpi=300)
    
    # plt.show()
    # plt.close()

    return np.max(z), np.min(z), z[-1].item(), z[0].item()

def execute_experiment(path, dynamics, experiment_name, dt=0.01):

    ''' 
    
    Function to train the model with the data in the path and the dynamics given.
    The function returns the alpha and beta values of the model trained.
    
    Parameters:

        path: str
            Path of the data to train the model.
        dynamics: str
            Dynamics to train the model.
        dt: float
            Time step of the data.
    
    Returns:

        latentEncoder_I: torch model
            Trained model.
        [alpha, beta, max_z, min_z]]: list
            List with the alpha and beta values of the model trained.

    Example:
        
            alpha, beta = execute_experiment('Data/data.npy', 'lorenz', 0.01)

    
    '''

    torch.cuda.empty_cache() 
    torch.manual_seed(42)

    global video_number 

    data_folder = np.load(path, allow_pickle=True)
    data_train = data_folder

    sample_frames = data_train[0,0,:,:]

    # plot and save the first frame

    for i in range(1, data_train.shape[1]):
        sample_frames = np.hstack((sample_frames, data_train[0,i,:,:]))
    
    sample_frames = sample_frames.transpose(1,2,0)
    plt.figure(figsize=(5,25))
    plt.imshow(sample_frames, cmap= 'gray')
    plt.savefig(f'./Results/{experiment_name}/sample_frame.png', dpi=300)
    plt.show()
    plt.close()
  
    print("Data shape: ", data_train.shape)
    print("Data range: \nMin: ", np.min(data_train), "\nMax: ", np.max(data_train)) 
    
    if data_train.shape[0] < 16:
        batch_size = 4
    elif data_train.shape[0] <32:
        batch_size = 8
    elif data_train.shape[0] < 64:
        batch_size = 16
    elif data_train.shape[0] < 128:
        batch_size = 32
    elif data_train.shape[0] < 256:
        batch_size = 64
    else:
        batch_size = 128
    
    batch_size = 128

    train_dataloader, test_dataloader, train_x, val_x  = loader.getLoader_folder(data_train, split=True, batch_size = batch_size)


    #define the model

    latentEncoder_I = mainmodel.EndPhys(dt = dt,
                                    pmodel = dynamics,
                                    init_phys = 10.0, 
                                    initw=True)

    #train model
    latentEncoder_I, log, params  = train.train(latentEncoder_I, 
                                    train_dataloader, 
                                    test_dataloader,
                                    lr_phys = 0.01,                                 
                                    loss_name='latent_loss',
                                    experiment_name=experiment_name)
    
    alpha_log = []
    alpha_log.append( [element["alpha"] for element in log  ])

    #plot alpha log
    plt.figure(figsize=(10,5))
    plt.plot(alpha_log[0], label='alpha')
    plt.xlabel('Epoch')
    plt.ylabel('Alpha')
    plt.legend()
    plt.savefig(f'./Results/{experiment_name}/alpha_log_{video_number}.png', dpi=300)
    plt.show()
    plt.close()

    
    
    checkpoint = torch.load(f'./Results/{experiment_name}/best_model.pt', weights_only=True)
    latentEncoder_I.load_state_dict(checkpoint)
    
    alpha = latentEncoder_I.pModel.alpha[0].detach().cpu().numpy().item()
    beta = latentEncoder_I.pModel.beta[0].detach().cpu().numpy().item()

    max_z, min_z, z0, z1 = evaluate_model(latentEncoder_I, data_train, dt, experiment_name+'/'+str(video_number))

    return latentEncoder_I, [alpha, beta, max_z, min_z, z0, z1]

def get_dynamics(path):
    ''' 
    Function to get the dynamics of the model trained.
    
    Parameters:
        path (str): The file path to check for specific dynamics keywords.

    Returns:
        str: The dynamic type found in the path, if any, else None.
    '''	
    dynamics_keywords = ['pendulum', 'sliding_block', 'bouncing_ball', 'dropped_ball', 'led', 'free_fall', 'torricelli' ]  # Add more keywords as needed

    # Normalize the path to ignore case and spaces
    normalized_path = path.replace(' ', '').lower()

    for keyword in dynamics_keywords:
        if keyword in normalized_path:
            print(f"Found dynamics keyword: {keyword}")
            return keyword
    
    return None

def iterate_folders_and_process(root_folder, output_folder='output', dt = 0.01):

    global video_number 

    # Initialize a list to collect data
    results = []

    # Iterate through all folders in the root directory
    for folder_name, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(folder_name, file)
                
                # Extract folder path components relative to the root folder
                relative_folder_path = os.path.relpath(folder_name, root_folder)
                path_components = relative_folder_path.split(os.sep)

                print(f"Processing {file_path}...")

                try:
                    # Execute the experiment
                    experiment = '_'.join(component.replace(' ', '') for component in path_components)
                    current_dynamics = get_dynamics(file_path)
                    print(f"Current dynamics: {current_dynamics}")
                    video_number += 1
                    model, [a, b, max_z, min_z, z0, z1] = execute_experiment(file_path, current_dynamics,  output_folder, dt)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    a, b, max_z, min_z, z0, z1 = 0,0,0,0,0,0


                results.append(path_components + [a, b, max_z, min_z, z0, z1])

    # Write results to a CSV file
    max_depth = max(len(row) - 2 for row in results)  # Determine max depth of folder structure
    #headers = [f'Folder_Level_{i+1}' for i in range(max_depth)] + ['alpha', 'beta', 'max_z', 'min_z']
    headers = ['run', 'alpha', 'beta', 'max_z', 'min_z',  'z0', 'z1']

    if not os.path.exists(f'./Results/{output_folder}'):
        os.makedirs(f'./Results/{output_folder}')

    with open(f'./Results/{output_folder}/{output_folder}.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)  # Header row
        for row in results:
            padded_row = row[:-2] + [''] * (max_depth - len(row[:-2])) + row[-2:]
            writer.writerow(padded_row)
    
    # mean_a, std_a, mean_b, std_b  = get_mean_std_from_csv('./Results/'+output_csv)

    # print(f"Mean alpha: {mean_a}, Std alpha: {std_a}")
    # print(f"Mean beta: {mean_b}, Std beta: {std_b}")

    print(f"Data successfully written to {output_folder}")

def main():
    '''
    Main function to execute the experiment.

    Parameters:

        None

    Returns:

        None

    Example:

    python main.py --path Data/data.npy --experiment_name experiment1 --dynamics lorenz --dt 0.01

    '''
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU in use: {gpu_name}")

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU found or GPU is not being utilized.")
        

    parser = argparse.ArgumentParser(description="Required parameter for a single experiment")
    #Example 
    #python main.py --path Data/data.npy --experiment_name experiment1 --dynamics lorenz --dt 0.01

    # Adding arguments
    
    parser.add_argument("--dt", type=str, required=True, help="Delta time")
    parser.add_argument("--path", type=str, required=True, help="Data path")
    parser.add_argument("--outfolder", type=str, default="output.csv", help="Output CSV file name")
    args = parser.parse_args()

    # Evaluate the expression safely
    dt = eval(args.dt)

    # Call your function with the specified or default output file name
    iterate_folders_and_process(args.path, output_folder=args.outfolder,dt = dt )

def get_mean_std_from_csv(csv_file):
    '''
    Function to get the mean and standard deviation of the data in the csv file.

    Parameters:

        csv_file: str
            Path of the csv file.
    
    Returns:

        mean: float
            Mean of the data in the csv file.
        std: float
            Standard deviation of the data in the csv file.

    Example:

        mean, std = get_mean_std_from_csv('output.csv')

    '''
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    a = data[:,1]
    b = data[:,2]
    mean_a = np.mean(data[:,1])
    std_a = np.std(data[:,1])
    mean_b = np.mean(data[:,2])
    std_b = np.std(data[:,2])

    return mean_a, std_a, mean_b, std_b

if __name__ == "__main__":

    '''
    Main function to execute the experiment.

    Parameters:
    
            None

    Returns:
    
                None
    Example:

    python main.py --path Data/data.npy --experiment_name experiment1 --dynamics lorenz --dt 0.01
    CUDA_VISIBLE_DEVICES=0,1 apptainer exec --nv /home/acastanedagarc/Projects/Vphy/Vphy/container_vphys.sif python /home/acastanedagarc/Projects/Vphy/Vphy/main.py --path /home/acastanedagarc/Projects/data/ --dt 0.1 >> output.log 2>&1
    '''
    
    torch.cuda.empty_cache() 
    torch.manual_seed(42)
    main()
