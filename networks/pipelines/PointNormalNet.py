from .normal_estimation.code.model import PointCloudNet




class PointNormalNet():
    def __init__(self,**argv):
        self.pipeline = PointCloudNet(**argv)
        
    def get_pipeline(self):
        return self.pipeline