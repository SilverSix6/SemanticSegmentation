import cv2

def save_image(file_dir, file_name, image):
    print(f'Image saved: {file_dir}/{file_name}')
    cv2.imwrite(f'{file_dir}/{file_name}', image)