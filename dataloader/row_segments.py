
"""
This file contains the row segments and rotation angles for the different datasets.
The rows are segmented in the following way:
        First, the path is rotated to align the rows with the axis. 
        Then, each row is segmented using a retangle-like bounding box. 

The row segments are defined as a dictionary with the following structure:

<dataset_name> = 
        {'angle': <angle>, # Angle to aline the rows with the axis
         'rows':
                [  
                [xmin,xmax,ymin,ymax], # row 1
                [xmin,xmax,ymin,ymax], # row 2
                ...
                [xmin,xmax,ymin,ymax]  # row n
                ]
        }

"""

kitti_00 = {'angle':0,  # GT ORCHARDS 10nov23_00
        'rows':
            [ 
             [ -10000,10000,-10000, 10000],
             
             ]
 } 
kitti_02 = {'angle':0,  # GT ORCHARDS 10nov23_00
        'rows':
            [ 
             [ -10000,10000,-10000, 10000],
             
             ]
 }

kitti_05 = {'angle':0,  # GT ORCHARDS 10nov23_00
        'rows':
            [[ -10000,10000,-10000, 10000]]
        }
kitti_06 = {'angle':0,  # GT ORCHARDS 10nov23_00
        'rows':
            [[ -10000,10000,-10000, 10000]]
        }
    
kitti_08 = {'angle':0,  # GT ORCHARDS 10nov23_00
        'rows':
            [[ -10000,10000,-10000, 10000]]
        }
  

ON23 = {'angle':-112.5,  # GT ORCHARDS 10nov23_00
        'rows':
            [ 
             #[ -55,-44,-1, 100],
             [ -25,-17,-3, 90],
             [ -17,-13,-3, 90],
             [ -13,-7,3, 90],
             [-25,5,-2,3],
             [-25,5,90,100],
             [ -7, -4,3, 90],
             [ -4, 5,3, 90],
             
             ]
 } 


ON22 = {'angle':1, # Angle to aline the rows with the image frame
        'rows':
            [ 
             [ -15,-9,-49, -1],
             [ -9,-5,-49, -1],
             [  -5,-2,-49, -1],
             [ -15, 5,-55,-49],
             [ -15, 5, -1,  5], 
             [  -2, 5,-49, -1],
             ]
 } 


OJ22 = {'angle':87, # -3, # Angle to aline the rows with the image frame
        'rows':
            [
            [ -8,-6,-39,-1],#, # row 3
            [ -6,-2,-39,-1],  # row 2
            [-2,2, -39,-1], # row 1
            [-8,2,-45,-39], # row 4
            [-8,2,-1,5]
            ]# row 5
 } 


GTJ23 = {'angle':88, # Angle to aline the rows with the image frame
        'rows':
        [[-1,0.5,1,37],
         [1,3,1,37],
         [4,7,1,37],
         [-2,8,-1,2],
         [-2,8,37,40]]
         }


OJ23 = {'angle':27, # Angle to aline the rows with the image frame
        'rows': # From left to right
        [[-5.5,-2.5,-45,0], # first vertial row
         [-2.5,2.5,-45,0], # second vertical row
         [2.5,5.5,-45,0],  # third vertical row
         [-5.5, 5.5,-50,-45], # bottom horizontal row
         [-5.5, 5.5, 0,10]]   # top horizontal row
         }

SJ23 = {'angle':26, # Angle to aline the rows with the image frame
        'rows':
        [[2,5,-105,-2],
         [10,15,-105,-2],
         [17,22,-105,-2],
         [2, 25, -120,-105],
         [0, 25,-2,5]]
         }


