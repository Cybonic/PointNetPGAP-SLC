
import numpy as np
import torch
from torch import nn


       
    
class SCANCONTEXT(nn.Module):
    def __init__(self,max_length=80, ring_res=20, sector_res=60,lidar_height=2.0,**argv):
        super(SCANCONTEXT,self).__init__()
 
        # static variables 
        self.lidar_height = lidar_height
        self.sector_res   = sector_res
        self.ring_res     = ring_res
        self.max_length   = max_length
      
    def xy2theta(self, x, y):
        if (x >= 0 and y >= 0): 
            theta = 180/np.pi * np.arctan(y/x)
        if (x < 0 and y >= 0): 
            theta = 180 - ((180/np.pi) * np.arctan(y/(-x)))
        if (x < 0 and y < 0): 
            theta = 180 + ((180/np.pi) * np.arctan(y/x))
        if ( x >= 0 and y < 0):
            theta = 360 - ((180/np.pi) * np.arctan((-y)/x))

        return theta
            
        
    def pt2rs(self, point, gap_ring, gap_sector, num_ring, num_sector):
        x = point[0]
        y = point[1]
        z = point[2]
        
        if(x == 0.0):
            x = 0.001
        if(y == 0.0):
            y = 0.001
     
        theta = self.xy2theta(x, y)
        faraway = np.sqrt(x*x + y*y)
        
        idx_ring = np.divmod(faraway, gap_ring)[0]       
        idx_sector = np.divmod(theta, gap_sector)[0]

        if(idx_ring >= num_ring):
            idx_ring = num_ring-1 # python starts with 0 and ends with N-1
        
        return int(idx_ring), int(idx_sector)
    
    
    def ptcloud2sc(self, ptcloud, num_sector, num_ring, max_length):
        
        num_points = ptcloud.shape[0]
        gap_ring = max_length/num_ring
        gap_sector = 360/num_sector
        
        enough_large = 1000
        sc_storage = np.zeros([enough_large, num_ring, num_sector])
        sc_counter = np.zeros([num_ring, num_sector])
        
        for pt_idx in range(num_points):

            point = ptcloud[pt_idx, :]
            point_height = point[2] + self.lidar_height
            
            idx_ring, idx_sector = self.pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)
            
            if sc_counter[idx_ring, idx_sector] >= enough_large:
                continue

            sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
            sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

        sc = np.amax(sc_storage, axis=0)
            
        return sc

    def __str__(self):
        return "scancontext"
    
    def forward(self,xyz):
        points = xyz.cpu().numpy().squeeze()
        sc = self.ptcloud2sc(points, self.sector_res, self.sector_res, self.max_length)
        sc = torch.tensor(sc).unsqueeze(dim=0)
        return sc
        

        
def similarity(sc1, sc2):
    num_sectors = sc1.shape[1]
    # repeate to move 1 columns
    sim_for_each_cols = np.zeros(num_sectors)

    for i in range(num_sectors):
        # Shift
        one_step = 1 # const
        sc1 = np.roll(sc1, one_step, axis=1) #  columne shift

        #compare
        sum_of_cos_sim = 0
        num_col_engaged = 0

        for j in range(num_sectors):
            col_j_1 = sc1[:, j]
            col_j_2 = sc2[:, j]

            if (~np.any(col_j_1) or ~np.any(col_j_2)):
                continue

            # calc sim
            cos_similarity = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
            sum_of_cos_sim = sum_of_cos_sim + cos_similarity

            num_col_engaged = num_col_engaged + 1

        # devided by num_col_engaged: So, even if there are many columns that are excluded from the calculation, we
        # can get high scores if other columns are well fit.
        sim_for_each_cols[i] = sum_of_cos_sim / num_col_engaged

    sim = max(sim_for_each_cols)
    dist = 1 - sim
    return dist
        
        
        
