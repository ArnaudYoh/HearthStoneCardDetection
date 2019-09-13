from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision

# The implementation is based around https://arxiv.org/pdf/1512.02325.pdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"


class PartialSSD300(nn.Module):
    """
    Encapsulates our partial VGG16 network and the prediction layers.
    """

    def __init__(self, n_classes):
        super(PartialSSD300, self).__init__()

        self.n_classes = n_classes

        self.base = PartialVGG16()
        self.pred_convs = PredictionLayers(n_classes)

        # Since lower level features have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in our feature map
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_centers = self.get_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: locations and class scores (i.e. w.r.t each prior box) for each image
        """

        feature_map = self.base(image)

        # Rescale the feature map with L2 norm
        norm = feature_map.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        feature_map = feature_map / norm  # (N, 512, 38, 38)
        feature_map = feature_map * self.rescale_factors  # (N, 512, 38, 38)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(feature_map)

        return locs, classes_scores

    def get_prior_boxes(self):
        """
        Create the prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes with center point and size (center_x, center_y, length on x axis, length on y axis)
        """

        # These values are taken from a https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
        fmap_dim = 38
        scale = 0.1
        aspect_ratios = [1., 2., 0.5]
        #aspect_ratios = [1., 2., 0.5, 3., 0.3]

        prior_boxes = list()
        for i in range(fmap_dim):
            for j in range(fmap_dim):
                center_x = (j + 0.5) / fmap_dim
                center_y = (i + 0.5) / fmap_dim

                for a_r in aspect_ratios:
                    prior_boxes.append([center_x, center_y, scale * sqrt(a_r), scale / sqrt(a_r)])

                    # Add a large box for large objects
                    if a_r == 1.:
                        additional_scale = 1. # This is empirical, I decided on it myself based on results
                        prior_boxes.append([center_x, center_y, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        # Taken from existing implementation
        """
        Decipher the locations and class scores (output of the SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes
        whose max class has a prediction score above min_score
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_centers.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            box_coordinate = center_coord_to_xy(
                diff_with_prior_coord_to_center_coord(predicted_locs[i], self.priors_centers))

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = box_coordinate[score_above_min_score]

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

class PartialVGG16(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(PartialVGG16, self).__init__()

        # Part of the Layers used in VGG16
        # Based on the SSD300 paper, the first feature map is the one that is best used to
        # analyse smaller objects. Given that each card in the images are usually very small
        # (less than 10% of the image surface), we decided to only keep the first feature map.
        # This way, we avoid unuseful computation and increasing the training speed.
        self.network = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),

                                     nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),

                                     nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),

                                     nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True)
                                     )

        self.load_pretrained_layers()

    def forward(self, image):
        out = self.network(image)
        feature_map = out

        return feature_map

    def load_pretrained_layers(self):
        """
        This allows to make use of transfer learning and avoid the need of ten tousands label images.
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        """

        # Pretrained VGG16
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)

        print("Params Loaded.\n")


class PredictionLayers(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes with our partialVGG16 output.
    We predict the boxes as encoded offsets from the priors.
    The class scores represent the scores of each object class in each of the bounding boxes located.
    """

    def __init__(self, n_classes):
        super(PredictionLayers, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering for the feature map
        self.n_boxes = 4

        # Box prediction convolutions (predict offsets w.r.t prior-boxes)
        self.box_pred = nn.Conv2d(512, self.n_boxes * 4, kernel_size=3, padding=1)
        # Class prediction convolutions (predict classes in boxes)
        self.class_pred = nn.Conv2d(512, self.n_boxes * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_params()

    def init_params(self):
        # Taken from existing implementation
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, feature_map):
        """
        Forward propagation.

        :param feature_map: a tensor of dimensions (N, 512, 38, 38)
        :return: box and class scores for each image
        """
        batch_size = feature_map.size(0)

        # Predict boxes' bounds (as offsets w.r.t prior-boxes)
        pred_box = self.box_pred(feature_map)
        print(pred_box.shape)
        pred_box = pred_box.permute(0, 2, 3, 1).contiguous()
        pred_box = pred_box.view(batch_size, -1, 4)

        # Predict classes in boxes
        pred_class = self.class_pred(feature_map)
        pred_class = pred_class.permute(0, 2, 3, 1).contiguous()
        pred_class = pred_class.view(batch_size, -1, self.n_classes)

        return pred_box, pred_class



class ModelLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    The Loss is a weighted sum:
    -  Box loss for the predicted locations of the boxes
    -  CLass loss for the predicted class scores.
    """

    def __init__(self, priors_centers, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(ModelLoss, self).__init__()
        self.priors_centers = priors_centers
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.threshold = threshold
        self.box_loss_fn = nn.L1Loss()
        self.class_loss_fn = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation of the loss
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_centers.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_boxes = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = jaccard_overlap(boxes[i], center_coord_to_xy(self.priors_centers))

            # For each prior, find the object that has the maximum overlap
            prior_max_overlap, object_for_prior_max_overlap = overlap.max(dim=0)
            # For each object, find the prior that has the maximum overlap.
            _, prior_max_overlap_for_object = overlap.max(dim=1)

            # Assign each object to the corresponding maximum-overlap-prior.
            object_for_prior_max_overlap[prior_max_overlap_for_object] = torch.LongTensor(range(n_objects)).to(device)
            # BOXES
            true_boxes[i] = center_coord_to_diff_with_prior_coord(
                xy_to_center_coord(boxes[i][object_for_prior_max_overlap]), self.priors_centers)

            # To ensure these priors qualify, artificially give them an overlap of 1.
            prior_max_overlap[prior_max_overlap_for_object] = self.threshold + .1
            label_for_each_prior = labels[i][object_for_prior_max_overlap]
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[prior_max_overlap < self.threshold] = 0
            # CLASSES
            true_classes[i] = label_for_each_prior


        # Identify priors that are positive (object/non-background)
        accepted_priors = true_classes != 0
        # Number of positive and negative accepted priors per image
        # With the ratio at 3, we accept 3 times as many negative priors for the class loss computation
        n_positives = accepted_priors.sum(dim=1)  # (N)
        n_negatives = self.neg_pos_ratio * n_positives  # (N)

        # BOX LOSS
        # Localization loss is computed only over positive (non-background) priors
        box_loss = self.box_loss_fn(predicted_locs[accepted_priors], true_boxes[accepted_priors])

        # Class LOSS
        # Class loss is computed over accepted priors and the worst rejected priors in each image
        # We take the worst (neg_pos_ratio * n_positives) rejected priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image,
        # and also minimizes pos/neg imbalance

        # First, find the loss for all priors
        class_loss_all_priors = self.class_loss_fn(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        class_loss_all_priors = class_loss_all_priors.view(batch_size, n_priors)

        # We sort negative priors in each image in order of decreasing loss and take top n_negatives
        class_loss_neg = class_loss_all_priors.clone()
        class_loss_neg[accepted_priors] = 0.
        class_loss_neg, _ = class_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(class_loss_neg).to(device)
        hard_negatives = hardness_ranks < n_negatives.unsqueeze(1)

        class_loss_pos_sum = class_loss_all_priors[accepted_priors].sum()  # (sum(n_positives))
        class_loss_hard_neg_sum = class_loss_neg[hard_negatives].sum()  # (sum(n_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        class_loss = (class_loss_hard_neg_sum + class_loss_pos_sum) / n_positives.sum().float()

        return class_loss + self.alpha * box_loss
