def load_yolo_annotations(annotation_path, img_width, img_height, return_labels=False):
    boxes = []
    label_map = {0: 'Good', 1: 'Bad', 2: 'Partially'}
    colors = {0: 'green', 1: 'red', 2: 'blue'}
    try:
        with open(annotation_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            label = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            if return_labels:
                boxes.append((x_min, y_min, x_max, y_max, label_map[label], colors[label]))
            else:
                boxes.append((x_min, y_min, x_max, y_max, label))
    except FileNotFoundError:
        print(f"❌ Annotation file not found -> {annotation_path}")
    return boxes
