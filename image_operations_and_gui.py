from PIL import Image, ImageTk   # used Python Imaging Library (PIL) modules
import numpy as np   # a fundamental package for scientific computing
import os   # os module is used for file and directory operations
# tkinter and its modules are used for creating a graphical user interface (gui)
import tkinter as tk
from tkinter import filedialog, messagebox

# Declaration and initialization of the global variables used in this program
# -------------------------------------------------------------------------------
# get the current directory where this program is placed
current_directory = os.path.dirname(os.path.realpath(__file__))
image_file_path = current_directory + '/thumbs_up.bmp'   # default image

# Main function where this program starts execution
# -------------------------------------------------------------------------------
def start():
   # create a window for the graphical user interface (gui)
   gui = tk.Tk() 
   # set the title of the window
   gui.title('Image Operations')
   # set the background color of the window
   gui['bg'] = 'SeaGreen1'
   # create and place a frame on the window with some padding for all four sides
   frame = tk.Frame(gui)
   # using the grid method for layout management
   frame.grid(row = 0, column = 0, padx = 15, pady = 15)
   # set the background color of the frame
   frame['bg'] = 'DodgerBlue4'
   # read and display the default image which is a thumbs up emoji
   gui_img = ImageTk.PhotoImage(file = image_file_path)
   gui_img_panel = tk.Label(frame, image = gui_img)
   # columnspan = 5 -> 5 columns as there are 5 buttons
   gui_img_panel.grid(row = 0, column = 0, columnspan = 5, padx = 10, pady = 10)
   # create and place five buttons below the image (button commands are expressed 
   # as lambda functions for enabling input arguments)
   # ----------------------------------------------------------------------------
   # the first button enables the user to open and view an image from a file
   btn1 = tk.Button(frame, text = 'Open Image', width = 10)
   btn1['command'] = lambda:open_image(gui_img_panel)
   btn1.grid(row = 1, column = 0) 
   # create and place the second button that shows the image in grayscale
   btn2 = tk.Button(frame, text = 'Grayscale', bg = 'gray', width = 10)
   btn2.grid(row = 1, column = 1)
   btn2['command'] = lambda:display_in_grayscale(gui_img_panel)
   # create and place the third button that shows the red channel of the image
   btn3 = tk.Button(frame, text = 'Red', bg = 'red', width = 10)
   btn3.grid(row = 1, column = 2)
   btn3['command'] = lambda:display_color_channel(gui_img_panel, 'red')
   # create and place the third button that shows the green channel of the image
   btn4 = tk.Button(frame, text = 'Green', bg = 'SpringGreen2', width = 10)
   btn4.grid(row = 1, column = 3)
   btn4['command'] = lambda:display_color_channel(gui_img_panel, 'green')
   # create and place the third button that shows the blue channel of the image
   btn5 = tk.Button(frame, text = 'Blue', bg = 'DodgerBlue2', width = 10)
   btn5.grid(row = 1, column = 4) 
   btn5['command'] = lambda:display_color_channel(gui_img_panel, 'blue')
   # wait for a gui event to occur and process each event that has occurred
   gui.mainloop()

# Function for opening an image from a file
# -------------------------------------------------------------------------------
def open_image(image_panel):
   global image_file_path   # to modify the global variable image_file_path
   # get the path of the image file selected by the user (.bmp is an uncompressed
   # image format that is suitable for this project)
   file_path = filedialog.askopenfilename(initialdir = current_directory, 
                                          title = 'Select an image file', 
                                          filetypes = [('bmp files', '*.bmp')])
   # display an warning message when the user does not select an image file
   if file_path == '':
      messagebox.showinfo('Warning', 'No image file is selected/opened.')
   # otherwise modify the global variable image_file_path and the displayed image
   else:
      image_file_path = file_path
      img = ImageTk.PhotoImage(file = image_file_path) 
      image_panel.config(image = img) 
      image_panel.photo_ref = img

# Function for displaying the current image in grayscale
# -------------------------------------------------------------------------------
def display_in_grayscale(image_panel):
   # open the current image as a PIL image
   img_rgb = Image.open(image_file_path)
   # convert the image to grayscale (img_grayscale has only one color channel 
   # whereas img_rgb has 3 color channels as red, green and blue)
   img_grayscale = img_rgb.convert('L')
   print('\nFor the color image')
   print('----------------------------------------------------------------------')
   width, height = img_rgb.size
   print('the width in pixels:', width, 'and the height in pixels:', height)
   img_rgb_array = pil_to_np(img_rgb)
   print('the dimensions of the image array:', img_rgb_array.shape)
   print('\nFor the grayscale image')
   print('----------------------------------------------------------------------')
   width, height = img_grayscale.size
   print('the width in pixels:', width, 'and the height in pixels:', height)
   img_grayscale_array = pil_to_np(img_grayscale)
   print('the dimensions of the image array:', img_grayscale_array.shape)
   # modify the displayed image
   img = ImageTk.PhotoImage(image = img_grayscale) 
   image_panel.config(image = img) 
   image_panel.photo_ref = img

# Function for displaying a given color channel of the current image
# -------------------------------------------------------------------------------
def display_color_channel(image_panel, channel):
   # red channel -> 0, green channel -> 1 and blue channel -> 2
   if channel == 'red':
      channel_index = 0
   elif channel == 'green':
      channel_index = 1
   else:
      channel_index = 2
   # open the current image as a PIL image
   img_rgb = Image.open(image_file_path)
   # convert the current image to a numpy array
   image_array = pil_to_np(img_rgb)
   # traverse all the pixels in the image array
   n_rows = image_array.shape[0]
   n_cols = image_array.shape[1]
   for row in range(n_rows):
      for col in range(n_cols):
         # make all the values 0 for the color channels except the given channel
         for rgb in range(3):
            if (rgb != channel_index):
               image_array[row][col][rgb] = 0
   # convert the modified image array (numpy) to a PIL image
   pil_img = np_to_pil(image_array)
   # modify the displayed image
   img = ImageTk.PhotoImage(image = pil_img) 
   image_panel.config(image = img) 
   image_panel.photo_ref = img

# Function that converts a given PIL image to a numpy array and returns the array
# -------------------------------------------------------------------------------
def pil_to_np(img):
   img_array = np.array(img)
   return img_array

# Function that converts a given numpy array to a PIL image and returns the image
# -------------------------------------------------------------------------------
def np_to_pil(img_array):
   img = Image.fromarray(np.uint8(img_array))
   return img

# start() function is specified as the entry point (main function) where this 
# program starts execution
if __name__== '__main__':
   start()