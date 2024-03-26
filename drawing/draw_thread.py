from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import pickle
import threading

## THIS FILE IS SOLELY FOR THREADING THE DRAW FUNCTION IN PLAYGROUND.IPYNB


def draw_and_save():
    # Create a new Tkinter window
    window = Tk()

    ## WIDTH AND HEIGHT EITHER 80 or 800
    wh = 80

    # Create a new canvas
    canvas = Canvas(window, width=wh, height=wh)
    canvas.pack()

    # Create a new PIL image and draw object
    image = Image.new("RGB", (8, 8), "white")
    draw = ImageDraw.Draw(image)

    def draw_pixel(event):
        x, y = event.x, event.y
        # Scale the coordinates to the size of the image
        x_image = x * 8 // wh
        y_image = y * 8 // wh
        canvas.create_rectangle(x, y, x+1, y+1, fill="black")
        # Set the pixel intensity to 16
        draw.point((x_image, y_image), fill=16)

    # Bind the left mouse button click event to the draw_pixel function
    canvas.bind("<B1-Motion>", draw_pixel)

    def save_image():
        global other_array
        # Resize the image to 8x8 pixels
        image_resized = image.resize((8, 8))
        # Convert the image to grayscale
        image_gray = image_resized.convert("L")
        # Normalize the pixel intensities to the range 0-16
        image_normalized = image_gray.point(lambda p:  p * 16 / 255)

        # Convert the image to a numpy array and print it
        image_array = np.array(image_normalized)
        other_array = np.array(image_normalized)
        for i, row in enumerate(image_array):
            for j, pixel in enumerate(row):
                if pixel == 0:
                    other_array[i][j] = 16
                else:
                    other_array[i][j] = 0
        # with open("image.pkl", "ab") as f:
        #     pickle.dump(other_array, f)
        window.quit()
        return other_array

        

    # Create a new button that will save the image when clicked
    button = Button(window, text="Save", command=save_image)
    button.pack()

    # Start the Tkinter event loop
    window.mainloop()

    return other_array

if __name__ == '__main__':
    draw_and_save()
