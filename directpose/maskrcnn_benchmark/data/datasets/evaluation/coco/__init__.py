from .coco_eval import do_coco_evaluation


def coco_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only=False,
    iou_types=["keypoints"],
    expected_results='eh2',
    expected_results_sigma_tol=20,
    datatype='bee',
):
    return do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        datatype=datatype,
    )
