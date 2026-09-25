import libs.data as data
from trackers import *

if __name__ == "__main__":
    cfg_file = "configs/siamdt_swin_tiny_sgd.py"
    name_suffix = cfg_file[8:-3]

    # Checkpoint produced by your training run
    ckp_file = "work_dirs/siamdt_swin_tiny_sgd/latest.pth"

    visualize = False
    selected_seq = "ALL"

    transforms = data.BasicPairTransforms(train=False)

    tracker = SiamDTTracker(
        cfg_file,
        ckp_file,
        transforms,
        name_suffix=name_suffix,
        visualize=visualize,
    )

    evaluators = [
        data.EvaluatorUAVtir(
            root_dir="/kaggle/input/datasets/uppalajathin/anti-uav-410-sample/",
            subset="test",
        )
    ]

    for e in evaluators:
        e.run(tracker, selected_seq=selected_seq)
