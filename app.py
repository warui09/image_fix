from bs4 import BeautifulSoup
import requests
import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
import numpy as np

os.makedirs('images', exist_ok=True)
os.makedirs('fixed_images', exist_ok=True)  # Create a folder for fixed images

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    processed_image_filenames = []  # List to store filenames of processed images

    if request.method == 'POST':
        url = request.form['url']
        processed_image_filenames = process_images(url)

    return render_template('index.html', processed_image_filenames=processed_image_filenames)

def process_images(url):
    images = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
        'Referer': 'https://www.google.com/',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8'
    }

    cont = requests.get(url, headers=headers).content
    soup = BeautifulSoup(cont, 'html.parser')
    imgall = soup.find_all('img')

    processed_image_filenames = []  # List to store filenames of processed images

    imgsdownloaded = 0
    imgsnotdownloaded = 0

    for img in imgall:
        try:
            imgsrc = img['data-srcset']
        except:
            try:
                imgsrc = img['data-src']
            except:
                try:
                    imgsrc = img['data-fallback-src']
                except:
                    try:
                        imgsrc = img['src']
                    except:
                        pass
        images.append(imgsrc)

    for image_url in images:
        if '.svg' in image_url:
            imgsnotdownloaded += 1
        else:
            r = requests.get(image_url).content
            filename = f'images/image{imgsdownloaded}.png'
            with open(filename, 'wb') as f:
                f.write(r)

            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            if img is not None:
                # Create a mask where pixels with low alpha values are marked as "invalid"
                alpha_threshold = 50  # Adjust this threshold as needed
                mask = (img[:, :, 3] < alpha_threshold).astype(np.uint8) * 255

                # Perform inpainting to fix blemishes
                inpainted_image = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

                # Save the processed image to the 'fixed_images' folder
                processed_filename = f'fixed_images/image{imgsdownloaded}_processed.png'
                cv2.imwrite(processed_filename, inpainted_image)

                # Add the processed image filename to the list
                processed_image_filenames.append(processed_filename)
                imgsdownloaded += 1

    return processed_image_filenames

if __name__ == '__main__':
    app.run(debug=True)
