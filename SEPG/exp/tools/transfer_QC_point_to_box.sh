export VERSION=1
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export SR=0.25
export CORNER=""
export WH=(12 12)
export T="train"

# tinyperson
# PYTHONPATH=. python /home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
#     "/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/data/TinyPerson/erase_with_uncertain_dataset/annotations/corner/task/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_tiny_set_train_sw640_sh512_all_coarse.json" \
#     "/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/data/TinyPerson/erase_with_uncertain_dataset/annotations/corner/task/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_tiny_set_train_sw640_sh512_all_coarse_with_gt.json"  \
#     --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}

# visdrone
PYTHONPATH=. python /home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
    "/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/data/VisDrone/VisDrone2019-DET-train/coco_fmt_annotations/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_visdrone_2019_coarse.json" \
    "/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/data/VisDrone/VisDrone2019-DET-train/coco_fmt_annotations/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_visdrone_2019_coarse_with_gt.json"  \
    --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}

# PYTHONPATH=. python /home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/huicv/coarse_utils/noise_data_mask_utils.py "generate_noisept_dataset" \
# "data/TinyPerson/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json" \
# "data/TinyPerson/erase_with_uncertain_dataset/annotations/corner/task/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_tiny_set_train_sw640_sh512_all_coarse.json" \
# --rand_type 'center_gaussian' --range_gaussian_sigma "(${MU[0]},${MU[1]})" --range_gaussian_sigma "(${S[0]},${S[1]})" \
# --size_range "${SR}"


