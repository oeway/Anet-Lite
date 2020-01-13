import os
from geojson_utils import masks_to_annotation, gen_mask_from_geojson

def generate_masks(datasets_dir):
    file_ids = os.listdir(os.path.join(datasets_dir, "train"))
    total = len(file_ids)
    for i, file_id in enumerate(file_ids):
        if file_id.startswith('.'):
            continue
        print(f'Processing {i}/{total}: {file_id}')
        file_path = os.path.join(datasets_dir, "train", file_id, "annotation.json")
        gen_mask_from_geojson([file_path], masks_to_create_value=["border_mask"], scale=4, img_size=(1024, 1024))

    print('Finished, generated masks saved to ' + datasets_dir)


if __name__ == '__main__':
    generate_masks('../SegmentationTraining_16bit_1024x1024')