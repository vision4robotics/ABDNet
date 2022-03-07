from .uavtrack112_l import UAVTrack112lDataset
from .uav10fps import UAV10Dataset
from .uav20l import UAV20Dataset
from .dtb import DTBDataset
from .uavdt import UAVDTDataset
from .visdrone1 import VISDRONED2018Dataset
from .v4r import V4RDataset
from .uav import UAVDataset
from .uav1231 import UAV123lDataset
from .dtb import DTBDataset
from .testreal import V4RtestDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):


        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        
        if 'UAV10' == name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV20' in name:
            dataset = UAV20Dataset(**kwargs)
        
        elif 'VISDRONED2018' in name:
            dataset = VISDRONED2018Dataset(**kwargs)
        elif'UAV101' in name :
            dataset = UAV123lDataset(**kwargs)
        elif 'UAVTrack112_l' in name:
            dataset = UAVTrack112lDataset(**kwargs)
        elif 'UAVTrack112' in name:
            dataset = V4RDataset(**kwargs)
        elif 'UAV123' in name:
            dataset = UAVDataset(**kwargs)
        elif 'DTB70' in name:
            dataset = DTBDataset(**kwargs)
        elif 'UAVDT' in name:
            dataset = UAVDTDataset(**kwargs)
        elif 'testreal' in name:
            dataset = V4RtestDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

