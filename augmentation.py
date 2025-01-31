import os

from matplotlib import pyplot as plt

from datasets.hico import *


def init_augmentation_methods(image_size: int):
    scales = [image_size - 48 * i for i in range(10) if (image_size - 32 * i) > 400]

    return Compose([
        RandomHorizontalFlip(),
        RandomAdjustImage(),
        RandomSelect(
            RandomResize(scales, max_size=2048),
            RandomSizeCrop(1200, 1800),
        ),
        ToTensor()
    ])

def get_class_specific_annotations(anno_path: str, interaction_class: str) -> [dict]:
    annotations = []
    for line in open(anno_path, "r").readlines():
        data = json.loads(line)
        object_info = data['gtboxes']
        for hoi in data["hoi"]:
            interaction = f"{hoi['interaction']} {object_info[hoi['object_id']]['tag']}"

            if interaction == interaction_class:
                annotations.append(data)

    annotations.sort(key=lambda x: x["file_name"])
    tensor_annotations = [parse_one_gt_line(json.dumps(anno)) for anno in annotations]

    return tensor_annotations

def display_augmentation_image(image_array_bgr, anno):
    for target in anno:
        # Draw all human_boxes
        if 'human_boxes' in target:
            for box in target['human_boxes']:
                x_min, y_min, x_max, y_max = box.tolist()
                cv2.rectangle(image_array_bgr,
                              (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)),
                              (0, 0, 255), 4)  # Red for human_boxes
        # Draw all object_boxes
        if 'object_boxes' in target:
            for box in target['object_boxes']:
                x_min, y_min, x_max, y_max = box.tolist()
                cv2.rectangle(image_array_bgr,
                              (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)),
                              (255, 0, 0), 4)  # Blue for object_boxes
    image_with_rectangles_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_array_bgr.shape
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(image_with_rectangles_rgb)
    plt.axis('off')
    plt.tight_layout(pad=0)

    plt.show()

def create_augmentations(image_path: str, anno_path: str, output_dir: str, number_of_augmentations: int, interaction_class: str, display_images: bool = False) -> None:
    annotations = get_class_specific_annotations(anno_path, interaction_class)

    augmented_hois = []
    for i in range(number_of_augmentations):
        random_int = random.randint(0, len(annotations) - 1)

        current_image_path = image_path + annotations[random_int]['image_id']
        current_annotation = annotations[random_int]['annotations']
        img = cv2.imread(current_image_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')  # convert to rgb format

        Augmentor = init_augmentation_methods(min(img.width, img.height))
        images, anno = Augmentor(img, current_annotation)

        device = torch.device("cuda")
        images = images.to(device)

        anno = [{k: v.to(device) for k, v in t.items() if k not in ['image_id']} for t in (anno,)]

        image_array = images.permute(1, 2, 0).cpu().numpy()  # Convert CxHxW -> HxWxC
        image_array = (image_array * 255).astype(np.uint8)  # Assuming input is in [0, 1] range

        # Convert RGB (matplotlib format) to BGR (OpenCV format)
        image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Save the image
        new_file_name = f"augmented_{i}_{annotations[random_int]['image_id']}"
        original_image_path = os.path.join(images_dir, new_file_name)
        cv2.imwrite(original_image_path, image_array_bgr)

        subject_id = 0
        new_gtboxes = []
        new_hois = []
        for target in anno:
            for i in range(len(target['human_boxes'])):
                human_label_id = target['human_labels'][i].item()
                human_label = coco_instance_ID_to_name[human_label_id]

                x_min, y_min, x_max, y_max = target['human_boxes'][i].tolist()
                human_box = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max- y_min + 1)]

                object_label_id = target['object_labels'][i].item()
                object_label = coco_instance_ID_to_name[object_label_id]

                x_min, y_min, x_max, y_max = target['object_boxes'][i].tolist()
                object_box = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max- y_min + 1)]

                action_label_id = target['action_labels'][i].item()
                action_label = hoi_interaction_names[action_label_id]

                new_gtboxes.extend([
                    {
                        "tag": human_label,
                        "box": human_box,
                    },
                    {
                        "tag": object_label,
                        "box": object_box
                    }])

                new_hois.append(
                    {
                        "subject_id": subject_id,
                        "interaction": action_label,
                        "object_id": subject_id + 1
                    }
                )

                subject_id += 2

        height, width, _ = image_array_bgr.shape
        new_data = {
            "file_name": new_file_name,
            "width": width,
            "height": height,
            "gtboxes": new_gtboxes,
            "hoi": new_hois
        }
        augmented_hois.append(new_data)

        if display_images:
            display_augmentation_image(image_array_bgr, anno)

    with open(output_dir + '/augmented.odgt', 'w') as file:
        for augmented_hoi in augmented_hois:
            file.write(json.dumps(augmented_hoi) + '\n')

if __name__ == "__main__":
    annotation_file = 'vis_aug.odgt'
    path_to_images = "wallmount_train2015/"
    target_interaction = "drink_with bottle"

    output_directory = f"odgt/{target_interaction}"

    random.seed(42)
    create_augmentations(path_to_images, annotation_file, output_directory, 20, target_interaction, True)

