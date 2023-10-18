from typing import List, Dict
from code_loader.contract.datasetclasses import PreprocessResponse

from data.kitti import Kitti


def subset_images() -> List[PreprocessResponse]:
    kitti_obj = Kitti()
    kitti_data: Dict[str, List[str]] = kitti_obj.get_kitti_data()
    return [PreprocessResponse(data=kitti_data['train'], length=len(kitti_data['train'])),
            PreprocessResponse(data=kitti_data['validation'], length=len(kitti_data['validation']))]

