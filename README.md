# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.
## program:
```
**STEP 1:**

# Load sunglasses image
sunglasses_image = cv2.imread('coolers.jpg', cv2.IMREAD_UNCHANGED)

# Check if the image is loaded correctly
if sunglasses_image is None:
    print("Error: Sunglasses image not loaded. Check the file path and format.")
else:
    print("Sunglasses image loaded successfully.")

**STEP 2:**

from PIL import Image

# Load the image using PIL to check transparency
sunglasses_image_pil = Image.open('coolers.jpg')

# Check if the image has an alpha channel
if sunglasses_image_pil.mode == 'RGBA':
    print("Sunglasses image has an alpha channel.")
else:
    print("Warning: Sunglasses image does not have an alpha channel.")

**STEP 3:**

import cv2
import numpy as np

# Load the passport photo and sunglasses image (without alpha channel)
passport_image = cv2.imread('me.jpg')
sunglasses_image = cv2.imread('coolers.jpg')

# Convert the sunglasses image to have an alpha channel (set it to fully opaque)
sunglasses_image_with_alpha = cv2.cvtColor(sunglasses_image, cv2.COLOR_BGR2BGRA)

# Set the alpha channel (transparency) to 255 (fully opaque)
sunglasses_image_with_alpha[:, :, 3] = 255

# Now proceed with the face and eye detection steps (same as before)

# Convert passport image to grayscale for face detection
gray = cv2.cvtColor(passport_image, cv2.COLOR_BGR2GRAY)

# Load pre-trained Haar cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    # Extract the region of interest for the eyes (in the face region)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = passport_image[y:y+h, x:x+w]
    
    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    # Assuming we found two eyes
    if len(eyes) == 2:
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[1]
        
        # Compute the width and height for the sunglasses overlay
        sunglass_width = ex2 + ew2 - ex1  # Difference in x-coordinates (eye width)
        sunglass_height = int(sunglasses_image_with_alpha.shape[0] * sunglass_width / sunglasses_image_with_alpha.shape[1])

        # Resize sunglasses image to match the eye region
        resized_sunglasses = cv2.resize(sunglasses_image_with_alpha, (sunglass_width, sunglass_height))
        
        # Get the region of interest for the sunglasses position
        sunglasses_region = roi_color[ey1:ey1+sunglass_height, ex1:ex1+sunglass_width]

        # Apply alpha blending to place the sunglasses on the image
        for i in range(0, sunglass_height):
            for j in range(0, sunglass_width):
                if resized_sunglasses[i, j][3] != 0:  # Check if the pixel is not transparent
                    sunglasses_region[i, j] = resized_sunglasses[i, j][:3]  # Apply RGB values

# Convert BGR image to RGB for display with matplotlib
output_image = cv2.cvtColor(passport_image, cv2.COLOR_BGR2RGB)

# Display the final image with sunglasses applied
import matplotlib.pyplot as plt
plt.imshow(output_image)
plt.axis('off')  # Hide axis
plt.show()

# Save the final image with sunglasses applied
cv2.imwrite('output_image_with_sunglasses.jpg', passport_image)

```
## output:

![Screenshot 2025-04-27 195526](https://github.com/user-attachments/assets/0503e818-8e6a-4e9f-ab2a-bad109ba7457)

