from . import BaseActor
import torch
import torch.nn.functional as F


def _float32(x):
    """Convert floating-point tensors to float32 without changing integer/bool tensors."""
    if torch.is_tensor(x) and torch.is_floating_point(x):
        return x.float()
    return x


def _float32_recursive(x):
    """Recursively convert floating-point tensors in nested structures to float32."""
    if torch.is_tensor(x):
        return _float32(x)

    if isinstance(x, dict):
        return {k: _float32_recursive(v) for k, v in x.items()}

    if isinstance(x, list):
        return [_float32_recursive(v) for v in x]

    if isinstance(x, tuple):
        return tuple(_float32_recursive(v) for v in x)

    return x


def _float32_data(data, keys):
    """Return selected data fields converted to float32 when they are floating tensors."""
    result = {}
    for key in keys:
        value = data[key]
        result[key] = _float32(value)
    return result


class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)

        # Keep model parameters/buffers in float32.
        self.net.float()

        if loss_weight is None:
            loss_weight = {"iou": 1.0, "test_clf": 1.0}

        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images',
                    'test_images', 'train_anno', 'test_proposals',
                    'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats   - dict containing detailed losses
        """

        train_images = _float32(data["train_images"])
        test_images = _float32(data["test_images"])
        train_anno = _float32(data["train_anno"])
        test_proposals = _float32(data["test_proposals"])
        test_label = _float32(data["test_label"])
        test_anno = _float32(data["test_anno"])
        proposal_iou = _float32(data["proposal_iou"])

        # Run network
        target_scores, iou_pred = self.net(
            train_imgs=train_images,
            test_imgs=test_images,
            train_bb=train_anno,
            test_proposals=test_proposals,
        )

        target_scores = _float32_recursive(target_scores)
        iou_pred = _float32(iou_pred)

        # Classification losses for the different optimization iterations
        clf_losses_test = [
            _float32(self.objective["test_clf"](s, test_label, test_anno))
            for s in target_scores
        ]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = _float32(self.loss_weight["test_clf"] * clf_loss_test)

        # Compute loss for ATOM IoUNet
        loss_iou = _float32(
            self.loss_weight["iou"] * self.objective["iou"](iou_pred, proposal_iou)
        )

        # Loss for the initial filter iteration
        loss_test_init_clf = 0.0
        if "test_init_clf" in self.loss_weight.keys():
            loss_test_init_clf = _float32(
                self.loss_weight["test_init_clf"] * clf_losses_test[0]
            )

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0.0
        if "test_iter_clf" in self.loss_weight.keys():
            test_iter_weights = self.loss_weight["test_iter_clf"]

            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = _float32(
                    sum(a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1]))
                )
            else:
                loss_test_iter_clf = _float32(
                    (test_iter_weights / (len(clf_losses_test) - 2))
                    * sum(clf_losses_test[1:-1])
                )

        # Total loss
        loss = _float32(
            loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf
        )

        if not torch.isfinite(loss).all():
            raise ValueError("NaN or Inf detected in loss")

        # Log stats
        stats = {
            "Loss/total": loss.item(),
            "Loss/iou": loss_iou.item(),
            "Loss/target_clf": loss_target_classifier.item(),
        }

        if "test_init_clf" in self.loss_weight.keys():
            stats["Loss/test_init_clf"] = loss_test_init_clf.item()

        if "test_iter_clf" in self.loss_weight.keys():
            stats["Loss/test_iter_clf"] = loss_test_iter_clf.item()

        stats["ClfTrain/test_loss"] = clf_loss_test.item()

        if len(clf_losses_test) > 0:
            stats["ClfTrain/test_init_loss"] = clf_losses_test[0].item()

            if len(clf_losses_test) > 2:
                stats["ClfTrain/test_iter_loss"] = sum(clf_losses_test[1:-1]).item() / (
                    len(clf_losses_test) - 2
                )

        return loss, stats


class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)

        self.net.float()

        if loss_weight is None:
            loss_weight = {"bb_ce": 1.0}

        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images',
                    'test_images', 'train_anno', 'test_proposals',
                    'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats   - dict containing detailed losses
        """

        train_images = _float32(data["train_images"])
        test_images = _float32(data["test_images"])
        train_anno = _float32(data["train_anno"])
        test_proposals = _float32(data["test_proposals"])
        test_label = _float32(data["test_label"])
        test_anno = _float32(data["test_anno"])
        proposal_density_all = _float32(data["proposal_density"])
        gt_density_all = _float32(data["gt_density"])

        # Run network
        target_scores, bb_scores = self.net(
            train_imgs=train_images,
            test_imgs=test_images,
            train_bb=train_anno,
            test_proposals=test_proposals,
        )

        target_scores = _float32_recursive(target_scores)
        bb_scores = _float32(bb_scores)

        # Reshape bb reg variables
        is_valid = test_anno[:, :, 0] < 99999.0

        bb_scores = bb_scores[is_valid, :]
        proposal_density = proposal_density_all[is_valid, :]
        gt_density = gt_density_all[is_valid, :]

        # Compute loss
        bb_ce = _float32(
            self.objective["bb_ce"](
                bb_scores,
                sample_density=proposal_density,
                gt_density=gt_density,
                mc_dim=1,
            )
        )

        loss_bb_ce = _float32(self.loss_weight["bb_ce"] * bb_ce)

        # If standard DiMP classifier is used
        loss_target_classifier = 0.0
        loss_test_init_clf = 0.0
        loss_test_iter_clf = 0.0

        if "test_clf" in self.loss_weight.keys():
            clf_losses_test = [
                _float32(self.objective["test_clf"](s, test_label, test_anno))
                for s in target_scores
            ]

            clf_loss_test = clf_losses_test[-1]

            loss_target_classifier = _float32(
                self.loss_weight["test_clf"] * clf_loss_test
            )

            if "test_init_clf" in self.loss_weight.keys():
                loss_test_init_clf = _float32(
                    self.loss_weight["test_init_clf"] * clf_losses_test[0]
                )

            if "test_iter_clf" in self.loss_weight.keys():
                test_iter_weights = self.loss_weight["test_iter_clf"]

                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = _float32(
                        sum(
                            a * b
                            for a, b in zip(test_iter_weights, clf_losses_test[1:-1])
                        )
                    )
                else:
                    loss_test_iter_clf = _float32(
                        (test_iter_weights / (len(clf_losses_test) - 2))
                        * sum(clf_losses_test[1:-1])
                    )

        # If PrDiMP classifier is used
        loss_clf_ce = 0.0
        loss_clf_ce_init = 0.0
        loss_clf_ce_iter = 0.0

        if "clf_ce" in self.loss_weight.keys():
            clf_ce_losses = [
                _float32(
                    self.objective["clf_ce"](
                        s, _float32(data["test_label_density"]), grid_dim=(-2, -1)
                    )
                )
                for s in target_scores
            ]

            clf_ce = clf_ce_losses[-1]

            loss_clf_ce = _float32(self.loss_weight["clf_ce"] * clf_ce)

            if "clf_ce_init" in self.loss_weight.keys():
                loss_clf_ce_init = _float32(
                    self.loss_weight["clf_ce_init"] * clf_ce_losses[0]
                )

            if "clf_ce_iter" in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight["clf_ce_iter"]

                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = _float32(
                        sum(
                            a * b
                            for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])
                        )
                    )
                else:
                    loss_clf_ce_iter = _float32(
                        (test_iter_weights / (len(clf_ce_losses) - 2))
                        * sum(clf_ce_losses[1:-1])
                    )

        # Total loss
        loss = _float32(
            loss_bb_ce
            + loss_clf_ce
            + loss_clf_ce_init
            + loss_clf_ce_iter
            + loss_target_classifier
            + loss_test_init_clf
            + loss_test_iter_clf
        )

        if not torch.isfinite(loss).all():
            raise ValueError("ERROR: Loss was NaN or Inf")

        # Log stats
        stats = {
            "Loss/total": loss.item(),
            "Loss/bb_ce": bb_ce.item(),
            "Loss/loss_bb_ce": loss_bb_ce.item(),
        }

        if "test_clf" in self.loss_weight.keys():
            stats["Loss/target_clf"] = loss_target_classifier.item()

        if "test_init_clf" in self.loss_weight.keys():
            stats["Loss/test_init_clf"] = loss_test_init_clf.item()

        if "test_iter_clf" in self.loss_weight.keys():
            stats["Loss/test_iter_clf"] = loss_test_iter_clf.item()

        if "clf_ce" in self.loss_weight.keys():
            stats["Loss/clf_ce"] = loss_clf_ce.item()

        if "clf_ce_init" in self.loss_weight.keys():
            stats["Loss/clf_ce_init"] = loss_clf_ce_init.item()

        if "clf_ce_iter" in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats["Loss/clf_ce_iter"] = loss_clf_ce_iter.item()

        if "test_clf" in self.loss_weight.keys():
            stats["ClfTrain/test_loss"] = clf_loss_test.item()

            if len(clf_losses_test) > 0:
                stats["ClfTrain/test_init_loss"] = clf_losses_test[0].item()

                if len(clf_losses_test) > 2:
                    stats["ClfTrain/test_iter_loss"] = sum(
                        clf_losses_test[1:-1]
                    ).item() / (len(clf_losses_test) - 2)

        if "clf_ce" in self.loss_weight.keys():
            stats["ClfTrain/clf_ce"] = clf_ce.item()

            if len(clf_ce_losses) > 0:
                stats["ClfTrain/clf_ce_init"] = clf_ce_losses[0].item()

                if len(clf_ce_losses) > 2:
                    stats["ClfTrain/clf_ce_iter"] = sum(clf_ce_losses[1:-1]).item() / (
                        len(clf_ce_losses) - 2
                    )

        return loss, stats


class KYSActor(BaseActor):
    """Actor for training KYS model."""

    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None):
        super().__init__(net, objective)

        self.net.float()

        self.loss_weight = loss_weight
        self.dimp_jitter_fn = dimp_jitter_fn

        # TODO set it somewhere
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, data):
        sequence_length = data["test_images"].shape[0]
        num_sequences = data["test_images"].shape[1]

        valid_samples = data["test_valid_image"].to(self.device).bool()
        test_visibility = _float32(data["test_visible_ratio"].to(self.device))

        # Initialize loss variables
        clf_loss_test_all = torch.zeros(
            num_sequences, sequence_length - 1, device=self.device, dtype=torch.float32
        )

        clf_loss_test_orig_all = torch.zeros(
            num_sequences, sequence_length - 1, device=self.device, dtype=torch.float32
        )

        dimp_loss_test_all = torch.zeros(
            num_sequences, sequence_length - 1, device=self.device, dtype=torch.float32
        )

        test_clf_acc = torch.zeros((), device=self.device, dtype=torch.float32)

        dimp_clf_acc = torch.zeros((), device=self.device, dtype=torch.float32)

        test_tracked_correct = torch.zeros(
            num_sequences, sequence_length - 1, device=self.device, dtype=torch.long
        )

        test_seq_all_correct = torch.ones(
            num_sequences, device=self.device, dtype=torch.float32
        )

        dimp_seq_all_correct = torch.ones(
            num_sequences, device=self.device, dtype=torch.float32
        )

        is_target_loss_all = torch.zeros(
            num_sequences, sequence_length - 1, device=self.device, dtype=torch.float32
        )

        is_target_after_prop_loss_all = torch.zeros(
            num_sequences, sequence_length - 1, device=self.device, dtype=torch.float32
        )

        # Initialize target model using the training frames
        train_images = _float32(data["train_images"].to(self.device))
        train_anno = _float32(data["train_anno"].to(self.device))

        dimp_filters = self.net.train_classifier(train_images, train_anno)

        dimp_filters = _float32_recursive(dimp_filters)

        # Track in the first test frame
        test_image_cur = _float32(data["test_images"][0, ...].to(self.device))

        backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)

        backbone_feat_prev_all = _float32_recursive(backbone_feat_prev_all)

        backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]

        backbone_feat_prev = backbone_feat_prev.view(
            1,
            num_sequences,
            -1,
            backbone_feat_prev.shape[-2],
            backbone_feat_prev.shape[-1],
        )

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all)

            motion_feat_prev = motion_feat_prev.view(
                1,
                num_sequences,
                -1,
                backbone_feat_prev.shape[-2],
                backbone_feat_prev.shape[-1],
            )
        else:
            motion_feat_prev = backbone_feat_prev

        motion_feat_prev = _float32(motion_feat_prev)

        dimp_scores_prev = self.net.dimp_classifier.track_frame(
            dimp_filters, backbone_feat_prev
        )

        dimp_scores_prev = _float32(dimp_scores_prev[:, :, :-1, :-1].contiguous())

        # Set previous frame information
        label_prev = _float32(data["test_label"][0:1, ...].to(self.device))

        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = _float32(data["test_anno"][0:1, ...].to(self.device))

        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1)

        # Loop over the sequence
        for i in range(1, sequence_length):
            test_image_cur = _float32(data["test_images"][i, ...].to(self.device))

            test_label_cur = _float32(
                data["test_label"][i : i + 1, ...].to(self.device)
            )

            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = _float32(data["test_anno"][i : i + 1, ...].to(self.device))

            # Extract features
            backbone_feat_cur_all = self.net.extract_backbone_features(test_image_cur)

            backbone_feat_cur_all = _float32_recursive(backbone_feat_cur_all)

            backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]

            backbone_feat_cur = backbone_feat_cur.view(
                1,
                num_sequences,
                -1,
                backbone_feat_cur.shape[-2],
                backbone_feat_cur.shape[-1],
            )

            if self.net.motion_feat_extractor is not None:
                motion_feat_cur = self.net.motion_feat_extractor(backbone_feat_cur_all)

                motion_feat_cur = motion_feat_cur.view(
                    1,
                    num_sequences,
                    -1,
                    backbone_feat_cur.shape[-2],
                    backbone_feat_cur.shape[-1],
                )
            else:
                motion_feat_cur = backbone_feat_cur

            motion_feat_cur = _float32(motion_feat_cur)

            # Run target model
            dimp_scores_cur = self.net.dimp_classifier.track_frame(
                dimp_filters, backbone_feat_cur
            )

            dimp_scores_cur = _float32(dimp_scores_cur[:, :, :-1, :-1].contiguous())

            # Jitter target model output for augmentation
            jitter_info = None

            if self.dimp_jitter_fn is not None:
                dimp_scores_cur = self.dimp_jitter_fn(
                    dimp_scores_cur, test_label_cur.clone()
                )

                dimp_scores_cur = _float32(dimp_scores_cur)

            # Input target model output along with previous frame information
            predictor_input_data = {
                "input1": motion_feat_prev,
                "input2": motion_feat_cur,
                "label_prev": label_prev,
                "anno_prev": anno_prev,
                "dimp_score_prev": dimp_scores_prev,
                "dimp_score_cur": dimp_scores_cur,
                "state_prev": _float32_recursive(state_prev),
                "jitter_info": _float32_recursive(jitter_info),
            }

            predictor_input_data = _float32_recursive(predictor_input_data)

            predictor_output = self.net.predictor(predictor_input_data)

            predictor_output = _float32_recursive(predictor_output)

            predicted_resp = _float32(predictor_output["response"])

            state_prev = predictor_output["state_cur"]

            aux_data = predictor_output["auxiliary_outputs"]

            is_valid = valid_samples[i, :].view(1, -1, 1, 1)

            visibility = test_visibility[i, :].view(1, -1, 1, 1)

            uncertain_frame = (visibility < 0.75) & (visibility > 0.25)

            is_valid = is_valid & ~uncertain_frame

            # Calculate losses
            clf_loss_test_new = _float32(
                self.objective["test_clf"](
                    predicted_resp,
                    test_label_cur,
                    test_anno_cur,
                    valid_samples=is_valid,
                )
            )

            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = _float32(
                self.objective["dimp_clf"](
                    dimp_scores_cur,
                    test_label_cur,
                    test_anno_cur,
                    valid_samples=is_valid,
                )
            )

            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if (
                "fused_score_orig" in aux_data
                and "test_clf_orig" in self.loss_weight.keys()
            ):
                aux_data["fused_score_orig"] = _float32(
                    aux_data["fused_score_orig"].view(test_label_cur.shape)
                )

                clf_loss_test_orig_new = _float32(
                    self.objective["test_clf"](
                        aux_data["fused_score_orig"],
                        test_label_cur,
                        test_anno_cur,
                        valid_samples=is_valid,
                    )
                )

                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()

            if (
                "is_target" in aux_data
                and "is_target" in self.loss_weight.keys()
                and "is_target" in self.objective.keys()
            ):
                is_target_loss_new = _float32(
                    self.objective["is_target"](
                        aux_data["is_target"], label_prev, is_valid_prev
                    )
                )

                is_target_loss_all[:, i - 1] = is_target_loss_new

            if (
                "is_target_after_prop" in aux_data
                and "is_target_after_prop" in self.loss_weight.keys()
                and "is_target" in self.objective.keys()
            ):
                is_target_after_prop_loss_new = _float32(
                    self.objective["is_target"](
                        aux_data["is_target_after_prop"], test_label_cur, is_valid
                    )
                )

                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective["clf_acc"](
                predicted_resp, test_label_cur, valid_samples=is_valid
            )

            test_clf_acc += _float32(test_clf_acc_new)

            test_seq_all_correct = (
                test_seq_all_correct
                * (test_pred_correct.long() | (~is_valid).long()).float()
            )

            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective["clf_acc"](
                dimp_scores_cur, test_label_cur, valid_samples=is_valid
            )

            dimp_clf_acc += _float32(dimp_clf_acc_new)

            dimp_seq_all_correct = (
                dimp_seq_all_correct
                * (dimp_pred_correct.long() | (~is_valid).long()).float()
            )

            motion_feat_prev = motion_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        # Compute average loss over the sequence
        clf_loss_test = _float32(clf_loss_test_all.mean())

        clf_loss_test_orig = _float32(clf_loss_test_orig_all.mean())

        dimp_loss_test = _float32(dimp_loss_test_all.mean())

        is_target_loss = _float32(is_target_loss_all.mean())

        is_target_after_prop_loss = _float32(is_target_after_prop_loss_all.mean())

        test_clf_acc = _float32(test_clf_acc / (sequence_length - 1))

        dimp_clf_acc = _float32(dimp_clf_acc / (sequence_length - 1))

        clf_loss_test_orig = _float32(clf_loss_test_orig / (sequence_length - 1))

        test_seq_clf_acc = _float32(test_seq_all_correct.mean())

        dimp_seq_clf_acc = _float32(dimp_seq_all_correct.mean())

        clf_loss_test_w = _float32(self.loss_weight["test_clf"] * clf_loss_test)

        clf_loss_test_orig_w = _float32(
            self.loss_weight["test_clf_orig"] * clf_loss_test_orig
        )

        dimp_loss_test_w = _float32(
            self.loss_weight.get("dimp_clf", 0.0) * dimp_loss_test
        )

        is_target_loss_w = _float32(
            self.loss_weight.get("is_target", 0.0) * is_target_loss
        )

        is_target_after_prop_loss_w = _float32(
            self.loss_weight.get("is_target_after_prop", 0.0)
            * is_target_after_prop_loss
        )

        loss = _float32(
            clf_loss_test_w
            + dimp_loss_test_w
            + is_target_loss_w
            + is_target_after_prop_loss_w
            + clf_loss_test_orig_w
        )

        if not torch.isfinite(loss).all():
            raise ValueError("NaN or Inf detected in loss")

        stats = {
            "Loss/total": loss.item(),
            "Loss/test_clf": clf_loss_test_w.item(),
            "Loss/dimp_clf": dimp_loss_test_w.item(),
            "Loss/raw/test_clf": clf_loss_test.item(),
            "Loss/raw/test_clf_orig": clf_loss_test_orig.item(),
            "Loss/raw/dimp_clf": dimp_loss_test.item(),
            "Loss/raw/test_clf_acc": test_clf_acc.item(),
            "Loss/raw/dimp_clf_acc": dimp_clf_acc.item(),
            "Loss/raw/is_target": is_target_loss.item(),
            "Loss/raw/is_target_after_prop": is_target_after_prop_loss.item(),
            "Loss/raw/test_seq_acc": test_seq_clf_acc.item(),
            "Loss/raw/dimp_seq_acc": dimp_seq_clf_acc.item(),
        }

        return loss, stats


class DiMPSimpleActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)

        self.net.float()

        if loss_weight is None:
            loss_weight = {"bb_ce": 1.0}

        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images',
                    'test_images', 'train_anno', 'test_proposals',
                    'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats   - dict containing detailed losses
        """

        train_images = _float32(data["train_images"])
        test_images = _float32(data["test_images"])
        train_anno = _float32(data["train_anno"])
        train_label = _float32(data["train_label"])
        test_anno = _float32(data["test_anno"])
        test_proposals = _float32(data["test_proposals"])
        proposal_density_all = _float32(data["proposal_density"])
        gt_density_all = _float32(data["gt_density"])
        test_label = _float32(data["test_label"])

        # Run network
        target_scores, bb_scores = self.net(
            train_imgs=train_images,
            test_imgs=test_images,
            train_bb=train_anno,
            test_proposals=test_proposals,
            train_label=train_label,
        )

        target_scores = _float32_recursive(target_scores)
        bb_scores = _float32(bb_scores)

        # Reshape bb reg variables
        is_valid = test_anno[:, :, 0] < 99999.0

        bb_scores = bb_scores[is_valid, :]
        proposal_density = proposal_density_all[is_valid, :]
        gt_density = gt_density_all[is_valid, :]

        # Compute loss
        bb_ce = _float32(
            self.objective["bb_ce"](
                bb_scores,
                sample_density=proposal_density,
                gt_density=gt_density,
                mc_dim=1,
            )
        )

        loss_bb_ce = _float32(self.loss_weight["bb_ce"] * bb_ce)

        loss_test_init_clf = 0.0
        loss_test_iter_clf = 0.0

        # Classification losses
        clf_losses_test = [
            _float32(self.objective["test_clf"](s, test_label, test_anno))
            for s in target_scores
        ]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]

        loss_target_classifier = _float32(self.loss_weight["test_clf"] * clf_loss_test)

        # Loss for initial filter iteration
        if "test_init_clf" in self.loss_weight.keys():
            loss_test_init_clf = _float32(
                self.loss_weight["test_init_clf"] * clf_losses_test[0]
            )

        # Loss for intermediate filter iterations
        if "test_iter_clf" in self.loss_weight.keys():
            test_iter_weights = self.loss_weight["test_iter_clf"]

            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = _float32(
                    sum(a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1]))
                )
            else:
                loss_test_iter_clf = _float32(
                    (test_iter_weights / (len(clf_losses_test) - 2))
                    * sum(clf_losses_test[1:-1])
                )

        # Total loss
        loss = _float32(
            loss_bb_ce
            + loss_target_classifier
            + loss_test_init_clf
            + loss_test_iter_clf
        )

        if not torch.isfinite(loss).all():
            raise ValueError("ERROR: Loss was NaN or Inf")

        stats = {
            "Loss/total": loss.item(),
            "Loss/bb_ce": bb_ce.item(),
            "Loss/loss_bb_ce": loss_bb_ce.item(),
        }

        if "test_clf" in self.loss_weight.keys():
            stats["Loss/target_clf"] = loss_target_classifier.item()

        if "test_init_clf" in self.loss_weight.keys():
            stats["Loss/test_init_clf"] = loss_test_init_clf.item()

        if "test_iter_clf" in self.loss_weight.keys():
            stats["Loss/test_iter_clf"] = loss_test_iter_clf.item()

        if "test_clf" in self.loss_weight.keys():
            stats["ClfTrain/test_loss"] = clf_loss_test.item()

            if len(clf_losses_test) > 0:
                stats["ClfTrain/test_init_loss"] = clf_losses_test[0].item()

                if len(clf_losses_test) > 2:
                    stats["ClfTrain/test_iter_loss"] = sum(
                        clf_losses_test[1:-1]
                    ).item() / (len(clf_losses_test) - 2)

        return loss, stats


class TargetCandiateMatchingActor(BaseActor):
    """Actor for training the KeepTrack network."""

    def __init__(self, net, objective):
        super().__init__(net, objective)

        self.net.float()

    def __call__(self, data):
        """
        args:
            data - The input data.

        returns:
            loss    - the training loss
            stats   - dict containing detailed losses
        """

        model_data = _float32_recursive(data)

        preds = self.net(**model_data)
        preds = _float32_recursive(preds)

        losses = self.objective["target_candidate_matching"](**model_data, **preds)

        losses = _float32_recursive(losses)

        # Total loss
        loss = _float32(losses["total"].mean())

        if not torch.isfinite(loss).all():
            raise ValueError("NaN or Inf detected in loss")

        # Log stats
        stats = {
            "Loss/total": loss.item(),
            "Loss/nll_pos": losses["nll_pos"].mean().item(),
            "Loss/nll_neg": losses["nll_neg"].mean().item(),
            "Loss/num_matchable": losses["num_matchable"].mean().item(),
            "Loss/num_unmatchable": losses["num_unmatchable"].mean().item(),
            "Loss/sinkhorn_norm": losses["sinkhorn_norm"].mean().item(),
            "Loss/bin_score": losses["bin_score"].item(),
        }

        if hasattr(self.objective["target_candidate_matching"], "metrics"):
            metrics = self.objective["target_candidate_matching"].metrics(
                **model_data, **preds
            )

            for key, val in metrics.items():
                val = _float32(val)

                valid = ~torch.isnan(val)

                if valid.any():
                    stats[key] = torch.mean(val[valid]).item()

        return loss, stats


class ToMPActor(BaseActor):
    """Actor for training the DiMP/ToMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)

        # Explicitly keep all model floating-point parameters/buffers
        # in float32.
        self.net.float()

        if loss_weight is None:
            loss_weight = {"bb_ce": 1.0}

        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        scores = _float32(scores)
        ltrb_gth = _float32(ltrb_gth)
        ltrb_pred = _float32(ltrb_pred)

        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]

        ids = scores.reshape(1, n, -1).max(dim=2)[1]

        index = torch.arange(0, n, device=scores.device)

        g = ltrb_gth.flatten(3)[0, index, :, ids].view(1, n, 4, 1, 1)

        p = ltrb_pred.flatten(3)[0, index, :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective["giou"](p, g)

        ious_pred_center = _float32(ious_pred_center)

        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images',
                    'test_images', 'train_anno', 'test_proposals',
                    'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats   - dict containing detailed losses
        """

        train_images = _float32(data["train_images"])
        test_images = _float32(data["test_images"])
        train_anno = _float32(data["train_anno"])
        train_label = _float32(data["train_label"])
        train_ltrb_target = _float32(data["train_ltrb_target"])

        test_ltrb_target = _float32(data["test_ltrb_target"])
        test_sample_region = _float32(data["test_sample_region"])
        test_label = _float32(data["test_label"])
        test_anno = _float32(data["test_anno"])

        # Run network
        target_scores, bbox_preds = self.net(
            train_imgs=train_images,
            test_imgs=test_images,
            train_bb=train_anno,
            train_label=train_label,
            train_ltrb_target=train_ltrb_target,
        )

        target_scores = _float32(target_scores)
        bbox_preds = _float32(bbox_preds)

        loss_giou, ious = self.objective["giou"](
            bbox_preds, test_ltrb_target, test_sample_region
        )

        loss_giou = _float32(loss_giou)
        ious = _float32(ious)

        # Classification loss
        clf_loss_test = self.objective["test_clf"](target_scores, test_label, test_anno)

        clf_loss_test = _float32(clf_loss_test)

        # Total loss
        loss = _float32(
            self.loss_weight["giou"] * loss_giou
            + self.loss_weight["test_clf"] * clf_loss_test
        )

        if not torch.isfinite(loss).all():
            raise ValueError("NaN or Inf detected in loss")

        ious_pred_center = self.compute_iou_at_max_score_pos(
            target_scores, test_ltrb_target, bbox_preds
        )

        stats = {
            "Loss/total": loss.item(),
            "Loss/GIoU": loss_giou.item(),
            "Loss/weighted_GIoU": self.loss_weight["giou"] * loss_giou.item(),
            "Loss/clf_loss_test": clf_loss_test.item(),
            "Loss/weighted_clf_loss_test": self.loss_weight["test_clf"]
            * clf_loss_test.item(),
            "mIoU": ious.mean().item(),
            "maxIoU": ious.max().item(),
            "minIoU": ious.min().item(),
            "mIoU_pred_center": ious_pred_center.mean().item(),
        }

        if ious.max().item() > 0:
            positive_ious = ious[ious > 0]

            if positive_ious.numel() > 1:
                stats["stdIoU"] = positive_ious.std().item()

        return loss, stats


class TaMOsActor(BaseActor):
    """Actor for training the TaMOs network."""

    def __init__(
        self,
        net,
        objective,
        loss_weight=None,
        prob=False,
        plot_save=None,
        fg_cls_loss=False,
    ):
        super().__init__(net, objective)

        self.net.float()

        if loss_weight is None:
            loss_weight = {"bb_ce": 1.0}

        self.loss_weight = loss_weight
        self.prob = prob
        self.plot_save = plot_save
        self.fg_cls_loss = fg_cls_loss

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        scores = _float32(scores)
        ltrb_gth = _float32(ltrb_gth)
        ltrb_pred = _float32(ltrb_pred)

        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]

        ids = scores.reshape(1, n, -1).max(dim=2)[1]

        index = torch.arange(0, n, device=scores.device)

        g = ltrb_gth.flatten(3)[0, index, :, ids].view(1, n, 4, 1, 1)

        p = ltrb_pred.flatten(3)[0, index, :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective["giou"](p, g)

        ious_pred_center = _float32(ious_pred_center)

        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def compute_cls_loss(
        self, test_label, target_scores, test_anno, test_sample_region
    ):
        test_label = _float32(test_label)
        target_scores = _float32(target_scores)
        test_anno = _float32(test_anno)
        test_sample_region = _float32(test_sample_region)

        test_label = test_label.flatten(1, 2)
        target_scores = target_scores.flatten(1, 2)
        test_sample_region = test_sample_region.flatten(1, 2)

        if self.fg_cls_loss:
            fg_mask = torch.sum(test_sample_region[0], dim=(1, 2)) > 0

            target_scores = target_scores[:, fg_mask]
            test_label = test_label[:, fg_mask]

        clf_loss_test = self.objective["test_clf"](target_scores, test_label, test_anno)

        return _float32(clf_loss_test)

    def compute_bbreg_loss(self, test_ltrb_target, test_sample_region, bbox_preds):
        test_ltrb_target = _float32(test_ltrb_target)
        test_sample_region = _float32(test_sample_region)
        bbox_preds = _float32(bbox_preds)

        test_ltrb_target = test_ltrb_target.flatten(1, 2)
        test_sample_region = test_sample_region.flatten(1, 2)
        bbox_preds = bbox_preds.flatten(1, 2)

        loss_giou, ious = self.objective["giou"](
            bbox_preds, test_ltrb_target, test_sample_region
        )

        return _float32(loss_giou), _float32(ious)

    def compute_iou_pred(
        self, target_scores, bbox_preds, test_ltrb_target, test_sample_region
    ):
        target_scores = _float32(target_scores)
        test_ltrb_target = _float32(test_ltrb_target)
        test_sample_region = _float32(test_sample_region)
        bbox_preds = _float32(bbox_preds)

        target_scores = target_scores.flatten(1, 2)
        test_ltrb_target = test_ltrb_target.flatten(1, 2)
        test_sample_region = test_sample_region.flatten(1, 2)
        bbox_preds = bbox_preds.flatten(1, 2)

        fg_mask = torch.sum(test_sample_region[0], dim=(1, 2)) > 0

        return self.compute_iou_at_max_score_pos(
            target_scores[:, fg_mask],
            test_ltrb_target[:, fg_mask],
            bbox_preds[:, fg_mask],
        )

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images',
                    'test_images', 'train_anno', 'test_proposals',
                    'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats   - dict containing detailed losses
        """

        train_images = _float32(data["train_images"])
        test_images = _float32(data["test_images"])
        train_anno = _float32(data["train_anno"])
        train_label = _float32(data["train_label"])
        train_ltrb_target = _float32(data["train_ltrb_target"])

        test_label = _float32(data["test_label"])

        # Run network
        target_scores, bbox_preds = self.net(
            train_imgs=train_images,
            test_imgs=test_images,
            train_bb=train_anno,
            train_label=train_label,
            train_ltrb_target=train_ltrb_target,
            test_label=test_label,
            epoch=data["epoch"],
        )

        target_scores = _float32_recursive(target_scores)
        bbox_preds = _float32_recursive(bbox_preds)

        stats = {}

        # Explicit float32 zero tensors.
        reference = train_images

        clf_loss_test = torch.zeros((), device=reference.device, dtype=torch.float32)

        loss_giou = torch.zeros((), device=reference.device, dtype=torch.float32)

        if "trafo" in target_scores:
            clf_loss_test_trafo = self.compute_cls_loss(
                test_label,
                target_scores["trafo"],
                _float32(data["test_anno"]),
                _float32(data["test_sample_region"]),
            )

            clf_loss_test = clf_loss_test + clf_loss_test_trafo

        if "lowres" in target_scores:
            clf_loss_test_lowres = self.compute_cls_loss(
                test_label,
                target_scores["lowres"],
                _float32(data["test_anno"]),
                _float32(data["test_sample_region"]),
            )

            clf_loss_test = clf_loss_test + clf_loss_test_lowres

        if "highres" in target_scores:
            clf_loss_test_highres = self.compute_cls_loss(
                _float32(data["test_label_highres"]),
                target_scores["highres"],
                _float32(data["test_anno"]),
                _float32(data["test_sample_region_highres"]),
            )

            clf_loss_test = clf_loss_test + clf_loss_test_highres

        if "trafo" in bbox_preds:
            loss_giou_trafo, ious_trafo = self.compute_bbreg_loss(
                _float32(data["test_ltrb_target"]),
                _float32(data["test_sample_region"]),
                bbox_preds["trafo"],
            )

            stats["mIoU_trafo"] = ious_trafo.mean().item()
            loss_giou = loss_giou + loss_giou_trafo

        if "lowres" in bbox_preds:
            loss_giou_lowres, ious_lowres = self.compute_bbreg_loss(
                _float32(data["test_ltrb_target"]),
                _float32(data["test_sample_region"]),
                bbox_preds["lowres"],
            )

            stats["mIoU_lowres"] = ious_lowres.mean().item()
            loss_giou = loss_giou + loss_giou_lowres

        if "highres" in bbox_preds:
            loss_giou_highres, ious_highres = self.compute_bbreg_loss(
                _float32(data["test_ltrb_target_highres"]),
                _float32(data["test_sample_region_highres"]),
                bbox_preds["highres"],
            )

            stats["mIoU_highres"] = ious_highres.mean().item()

            stats["Loss/weighted_GIoU_highres"] = (
                self.loss_weight["giou"] * loss_giou_highres.item()
            )

            loss_giou = loss_giou + loss_giou_highres

        if "trafo" in target_scores and "trafo" in bbox_preds:
            ious_pred_center_trafo = self.compute_iou_pred(
                target_scores["trafo"],
                bbox_preds["trafo"],
                _float32(data["test_ltrb_target"]),
                _float32(data["test_sample_region"]),
            )

            stats["mIoU_pred_center_trafo"] = ious_pred_center_trafo.mean().item()

        if "lowres" in target_scores and "lowres" in bbox_preds:
            ious_pred_center_lowres = self.compute_iou_pred(
                target_scores["lowres"],
                bbox_preds["lowres"],
                _float32(data["test_ltrb_target"]),
                _float32(data["test_sample_region"]),
            )

            stats["mIoU_pred_center_lowres"] = ious_pred_center_lowres.mean().item()

        if "highres" in target_scores and "highres" in bbox_preds:
            ious_pred_center_highres = self.compute_iou_pred(
                target_scores["highres"],
                bbox_preds["highres"],
                _float32(data["test_ltrb_target_highres"]),
                _float32(data["test_sample_region_highres"]),
            )

            stats["mIoU_pred_center_highres"] = ious_pred_center_highres.mean().item()

        if "highres" not in target_scores and "highres" in bbox_preds:
            if "trafo" in target_scores:
                target_scores_trafo = _float32(target_scores["trafo"].flatten(1, 2))

                target_scores_trafo_interp = F.interpolate(
                    target_scores_trafo,
                    _float32(data["test_ltrb_target_highres"]).shape[-2:],
                    mode="bicubic",
                )

                ious_pred_center_highres = self.compute_iou_pred(
                    target_scores_trafo_interp.reshape(
                        data["test_sample_region_highres"].shape
                    ),
                    bbox_preds["highres"],
                    _float32(data["test_ltrb_target_highres"]),
                    _float32(data["test_sample_region_highres"]),
                )

                stats["mIoU_pred_center_trafo_highres"] = (
                    ious_pred_center_highres.mean().item()
                )

            if "lowres" in target_scores:
                target_scores_lowres = _float32(target_scores["lowres"].flatten(1, 2))

                target_scores_lowres_interp = F.interpolate(
                    target_scores_lowres,
                    _float32(data["test_ltrb_target_highres"]).shape[-2:],
                    mode="bicubic",
                )

                ious_pred_center_highres = self.compute_iou_pred(
                    target_scores_lowres_interp.reshape(
                        data["test_sample_region_highres"].shape
                    ),
                    bbox_preds["highres"],
                    _float32(data["test_ltrb_target_highres"]),
                    _float32(data["test_sample_region_highres"]),
                )

                stats["mIoU_pred_center_lowres_highres"] = (
                    ious_pred_center_highres.mean().item()
                )

        # Final loss is explicitly float32.
        loss = _float32(
            self.loss_weight["giou"] * loss_giou
            + self.loss_weight["test_clf"] * clf_loss_test
        )

        if not torch.isfinite(loss).all():
            raise ValueError("NaN or Inf detected in loss")

        stats.update(
            {
                "Loss/total": loss.item(),
                "Loss/GIoU": loss_giou.item(),
                "Loss/weighted_GIoU": self.loss_weight["giou"] * loss_giou.item(),
                "Loss/clf_loss_test": clf_loss_test.item(),
                "Loss/weighted_clf_loss_test": self.loss_weight["test_clf"]
                * clf_loss_test.item(),
            }
        )

        return loss, stats
