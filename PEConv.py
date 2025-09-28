from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np
import cv2
import os

def get_points(p, n):
    hilbert_curve = HilbertCurve(p, n)
    points = np.array( hilbert_curve.points_from_distances( list(range(2 ** (p * n)))) )
    return points

def PEConverter(PE_path, points, output_path, IMAGE_SIZE, p, n):
    image = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    total_size = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    file_path = PE_path;
    with open(file_path, 'rb') as file:
        byte_data = file.read()    
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    byte_chunk = byte_array[0:total_size]

    for byte, index in zip(byte_chunk, list( range(2 ** (p * n)) ) ):
        color = tuple()
        if byte == 0:
            color = (0, 0, 0)    
        elif byte == 255:
            color = (255, 255, 255)
        elif 32 <= byte <= 126:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        image[points[index][1], points[index][0]] = color;
    cv2.imwrite(output_path, image)

def main():
    # order nine dimension two pseudo hilbert curve
    points = get_points(9, 2);

    directory = "/home/aqua/mount-file/temp-sda3/dataset-folder/Benign-NET"
    output_directory = "/home/aqua/mount-file/temp-sda3/dataset-folder/dataset"
    
    index = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".exe"):
                PEConverter(os.path.join(root, file), points, f"{output_directory}/{index}.jpg", (512, 512, 3), 9, 2)
                index += 1
    
    PEConverter("dataset/CNET_100032.exe", points, "dataset/PE-dataset/10.jpg", (512, 512, 3), 9, 2)
    
    
    
main();