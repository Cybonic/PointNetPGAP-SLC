
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


AUTUMN = {'angle': 0,  # Angle to aline the rows with the image frame
          'rows':
            [[-15,-9,-50, -1],
            [  -9,-5,-50, -1],
            [  -5,-2,-50, -1],
            [  -2, 2,-50, -1],
            [ -15, 2,-55,-49],
            [ -15, 2, -1,  5]]
        }

orchards_aut22 = {'angle':1, # Angle to aline the rows with the image frame
        'rows':
            [[ -15, 2,-55,-49],
             [ -15, 2, -1,  5],
             [-15,-10,-50, -1],
            [  -10,-5,-50, -1],
            [  -5,-2,-50, -1],
            [  -2, 5,-50, -1]
            ]
 } 


orchards_sum22 = {'angle':-3, # Angle to aline the rows with the image frame
        'rows':
            [
            [-39,-1, 4.7,8], # row 1
            [ -39,-1, 1.8,4],  # row 2
            [ -39,-1,-2,0.7],#, # row 3
            [-0.45,3,-2,8], # row 4
            [-45,-39,-2,8]
            ]# row 5
 } 


e3 = {'angle':-2, # Angle to aline the rows with the image frame
        'rows':
        [[-1,40,-0.4,2],
         [-1,40,-3.5,0.5],
         [-1,40,-6,-3.5],
         [-1, 2.5, -6,1],
         [35, 40,-6,1]]
         }


orchards_june23 = {'angle':25.5, # Angle to aline the rows with the image frame
        'rows': # From left to right
        [[-15,-12,-41,5], # first vertial row
         [-9,-6.7,-41,5], # second vertical row
         [-4.3,2,-41,5],  # third vertical row
         [-15, 4, -44,-41], # bottom horizontal row
         [-15, 4, 5,10]]   # top horizontal row
         }

strawberry_june23 = {'angle':24, # Angle to aline the rows with the image frame
        'rows':
        [[-30,-20,-108,-2],
         [-16,-10,-108,-2],
         [-7,-2,-108,-2],
         [-30, -2, -113,-108],
         [-30, -2,-2,5]]
         }


