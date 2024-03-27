
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

husky_orchards_10nov23_00 = {'angle':-112.5,  # GT ORCHARDS 10nov23_00
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


orchards_aut22 = {'angle':1, # Angle to aline the rows with the image frame
        'rows':
            [ 
             [ -15,-10,-50, -1],
             [ -10,-5,-50, -1],
             [  -5,-2,-50, -1],
             [ -15, 2,-55,-49],
             [ -15, 2, -1,  5], 
             [  -2, 5,-50, -1],
             ]
 } 


orchards_sum22 = {'angle':87, # -3, # Angle to aline the rows with the image frame
        'rows':
            [
            [ -39,-0,-2,0.7],#, # row 3
            [ -39,-0, 1.8,4],  # row 2
            [-39,-0, 4.7,8], # row 1
            [-1,3,-10,8], # row 4
            [-45,-39,-10,8]
            ]# row 5
 } 


e3 = {'angle':88, # Angle to aline the rows with the image frame
        'rows':
        [[-1,0.5,1,37],
         [1,3,1,37],
         [4,7,1,37],
         [-2,8,-1,2],
         [-2,8,37,40]]
         }


orchards_june23 = {'angle':27, # Angle to aline the rows with the image frame
        'rows': # From left to right
        [[-15,-11.5,-43,5], # first vertial row
         [-11.5,-7,-43,5], # second vertical row
         [-7,2,-43,5],  # third vertical row
         [-15, 4,-48,-43], # bottom horizontal row
         [-15, 4, 2,10]]   # top horizontal row
         }

strawberry_june23 = {'angle':26, # Angle to aline the rows with the image frame
        'rows':
        [[-30,-16,-108,-2],
         [-16,-10,-108,-2],
         [-10,-2,-108,-2],
         [-30, -2, -113,-103],
         [-30, -2,-2,5]]
         }


