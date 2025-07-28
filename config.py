import os

image_dir = ''
mask_dir = ''

classes = sorted(os.listdir(image_dir))

cls_to_int = {cls_name: idx for idx, cls_name in enumerate(classes)}
int_to_cls = {idx: cls_name for cls_name, idx in cls_to_int.items()}

print("Classes mapped to integers:", cls_to_int)