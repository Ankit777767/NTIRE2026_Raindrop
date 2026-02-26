import os
import cv2
import numpy as np
import glob

def create_median_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of all scene folders inside the input folder
    scene_folders = glob.glob(os.path.join(input_folder, '*'))
    
    for scene in scene_folders:
        # Make sure it is actually a folder
        if not os.path.isdir(scene):
            continue
            
        scene_id = os.path.basename(scene)
        image_paths = glob.glob(os.path.join(scene, '*.png'))
        
        # Skip empty folders
        if len(image_paths) == 0:
            continue
            
        print("Processing scene: " + scene_id + " with " + str(len(image_paths)) + " images")
        
        # Read all images in this scene folder
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                
        if len(images) == 0:
            continue
            
        # Stack images and calculate the median to remove moving raindrops
        stacked_images = np.stack(images, axis=0)
        median_image = np.median(stacked_images, axis=0).astype(np.uint8)
        
        # Convert to grayscale to check sharpness
        gray_image = cv2.cvtColor(median_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate the variance of the Laplacian to measure sharpness
        sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        # Save the image only if the background is sharp (threshold > 100)
        if sharpness > 100: 
            save_path = os.path.join(output_folder, "pseudo_gt_" + scene_id + ".png")
            cv2.imwrite(save_path, median_image)
            print(" -> Saved sharp background! (Sharpness score: " + str(round(sharpness, 2)) + ")")
        else:
            print(" -> Skipped. Background is too blurry. (Sharpness score: " + str(round(sharpness, 2)) + ")")

if __name__ == '__main__':
    # print("--- Starting Daytime ---")
    # # Change the paths if your dataset is located somewhere else
    # day_input = '/media/admin1/DL/ankit/raindrop/ntire2026-dualfocus-raindrop/data/train/daytime/Drop'
    # day_output = '/media/admin1/DL/ankit/raindrop/ntire2026-dualfocus-raindrop/data/train/daytime/pseudo_gt'
    # create_median_images(day_input, day_output)
    
    print("\n--- Starting VAL ---")
    night_input = '/media/admin1/DL/ankit/raindrop/ntire2026-dualfocus-raindrop/data/codabench(copy)'
    night_output = '/media/admin1/DL/ankit/raindrop/ntire2026-dualfocus-raindrop/data/codabench_val/pseudo_gt'
    create_median_images(night_input, night_output)
    
    print("\nDone generating pseudo ground truths!")