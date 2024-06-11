import os
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


SETTINGS = {'ON22': {'scale':250},
            'SJ23': {'scale':0.2},
            'OJ22': {'scale':10},
            'OJ23': {'scale':1},
}


def rgba_to_hex(rgba):
    r, g, b, a = rgba
    return "#{:02x}{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255), int(a*255))


def plot_3d_path(line_int, labels,samples=2000,vertical_scale=10.0):
    import matplotlib.pyplot as plt

    x = line_int[:,0]
    y = line_int[:,1]
    z = line_int[:,2]
    
    # Plot a subset of points, equally distributed
    num_points = samples  # Number of points to plot
    indices = np.linspace(0, len(x)-1, num_points, dtype=int)  # Equally spaced indices
    labels_subset = labels[indices]
    
    x_subset = x[indices]
    y_subset = y[indices]
    z_subset = vertical_scale*indices * (z[-1] - z[0]) / (len(x) - 1) + z[0]
    
    # Create a 2D path plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Colorize points based on label
    unique_labels = np.unique(labels_subset)
    colors = plt.cm.get_cmap('viridis', 7)
    #colors = plt.cm.get_cmap('viridis', len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels_subset == label
        print(f"Color {label}: {rgba_to_hex(colors(i))}")
        ax.scatter(x_subset[mask], y_subset[mask], z_subset[mask], color=colors(i), label=f'Label {label}')

    ax.axis('equal')  # Equal aspect ratio
    ax.set_axis_off()  # Turn off the axis
    plt.show()
    
def plot_2d_path(line_int, labels,samples=2000,file='path.png'):
    import matplotlib.pyplot as plt

    x = line_int[:,0]
    y = line_int[:,1]
    z = line_int[:,2]
    
    # Plot a subset of points, equally distributed
    num_points = samples  # Number of points to plot
    indices = np.linspace(0, len(x)-1, num_points, dtype=int)  # Equally spaced indices
    
    labels_subset = labels[indices]
    
    x_subset = x[indices]
    y_subset = y[indices]
    #z_subset = scaler*indices * (z[-1] - z[0]) / (len(x) - 1) + z[0]
    
    # Create a 2D path plot
    fig, ax = plt.subplots()
    
    # Colorize points based on label
    unique_labels = np.unique(labels_subset[labels_subset != -1])

    #colors = plt.colormaps('viridis', 7)
    colors = plt.cm.get_cmap('viridis', 7)
    
    
    for i, label in enumerate(unique_labels):
        mask = labels_subset == label
        ax.scatter(x_subset[mask], y_subset[mask], color=colors(i), label=f'Label {label}',s=10)

    ax.legend()
    # Set axis limits tight to the data points
    ax.set_aspect('equal')
    plt.savefig(file, dpi=300, bbox_inches='tight')  # Adjust the file format as needed
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Convert bag dataset to files!")
    parser.add_argument('--root', type=str, default='/home/tiago/workspace/DATASET', help='path to the data directory')
    parser.add_argument('--dataset',
                                    default = 'HORTO-3DLM',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    parser.add_argument('--seq',default  = "ON22",type = str, 
                        help='path to the data of the sequence')
    parser.add_argument('--pose_data_source',default  = "positions" ,type = str, choices = ['gps','poses'])
    parser.add_argument("--kitti_format",default=True,type=bool,help="Expects that poses.txt file to be kitti format")
    parser.add_argument("--plot3D",default=True,type=bool,help="Plot 3D path")
    args = parser.parse_args()

    print("RUNNING\n")
    target_dir = os.path.join(args.root,args.dataset,args.seq,'extracted')
    #target_dir = args.target_dir
    if not os.path.isdir(target_dir):
        print("target is not a directory")
        exit(0)

    parse_target_path = target_dir.split('/')
    # read files from target dir
    #files = os.listdir(target)
    pose_file = os.path.join(target_dir,args.pose_data_source+'.txt')
    assert os.path.isfile(pose_file), "File does not exist: " + pose_file
    
    
    if args.kitti_format :
        print("Kitti Format is On!")
        fd = open(pose_file,'r')
        lines = fd.readlines()

        print("Number of lines: " + str(len(lines)))
        line_int = []
        for line in lines:
            # map text to float transformation matrix
            coordinates = [float(value) for value in line.strip().split(' ')]

            if len(coordinates) == 16:
                coordinates = np.array(coordinates).reshape(4,4)
                #coordinates = np.append(coordinates,np.array([0,0,0,1])).reshape(4,4)
                line_int.append(coordinates[:3,3])
            elif len(coordinates) == 3:
                coordinates = np.array(coordinates)
                line_int.append(coordinates[:3])
            elif len(coordinates) == 7:
                coordinates = np.array(coordinates)
                line_int.append(coordinates[:3])
            else:    
                print("Error: Wrong format in kitti file")
                exit(0)
    # read segment labels
    label_file = os.path.join(target_dir,'point_row_labels.pkl')
    # read pickle file
    import pickle
    
    
    
    from mpl_toolkits.mplot3d import Axes3D
    with open(label_file, 'rb') as f:
        labels = pickle.load(f)
    print("Number of labels: " + str(len(labels)))
    print("Number of points: " + str(len(line_int)))
  
    
    # convert to numpy array
    line_int = np.array(line_int)#[:-700,:]
    
    if args.plot3D:
        plot_3d_path(line_int, labels,samples=2000,vertical_scale=SETTINGS[args.seq]['scale'])
    
    file_name = pose_file.split('.')[0]
    
    plot_2d_path(line_int, labels,samples=2000,file= f'{file_name}.png')
    print("DONE\n")
   

    
