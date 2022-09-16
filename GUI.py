# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:40:29 2022

@author: chonk
"""

#imports
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog

from tkinter.font import Font
from tkinter.filedialog import askopenfile

import PIL
from PIL import Image, ImageTk, ImageDraw

import Main
from Main import *

#%%
#window
def createWindow():
    global root
    model = create_network()
    root=Tk()
    root.title('Digit Recognition Tool')
    root.geometry("388x320")
    root.configure(bg='white')
    Upload, Create = createFrame(root, model)
    raiseFrame(Upload) #Displays the home frame first
    root.mainloop() #Displays the windows created

#simulation frame creation
def createFrame(root, model):  
    Upload = Frame(root)
    Create = Frame(root)
    
    for frame in (Upload, Create):
        frame.grid(row=0, column=0, sticky='news')
    
    #configuring geometry
    Upload.grid_columnconfigure(0, weight=5)
    Upload.grid_rowconfigure(0, weight=1)
    Upload.grid_rowconfigure(1, weight=1)
    Upload.grid_rowconfigure(2, weight=1)
    Upload.grid_rowconfigure(3, weight=1)
    Upload.grid_rowconfigure(4, weight=1)
    Upload.grid_rowconfigure(5, weight=1)

    #widgets 
    upload_button = ttk.Button(Upload, text='Upload File',  width=20, command=lambda:upload_file(Upload, model))
    upload_button.grid(row=1 ,column=0) 

    text1 = tk.Label(Upload, text="Please upload an image, sized 28x28 pixels, with a single numerical digit.")
    text1.grid(row=0, column=0)
    
    draw_button = ttk.Button(Upload, text='Draw instead',  width=20, command=lambda:raiseFrame(Create))
    draw_button.grid(row=2, column=0)
    
    #configuring geometry
    Create.grid_columnconfigure(0, weight=5)
    Create.grid_rowconfigure(0, weight=1)
    Create.grid_rowconfigure(1, weight=1)
    Create.grid_rowconfigure(2, weight=1)
    Create.grid_rowconfigure(3, weight=1)
    Create.grid_rowconfigure(4, weight=1)

    text3 = tk.Label(Create, text="Please draw a single numerical digit.")
    text3.grid(row=0, column=0)

    open_button = ttk.Button(Create, text='Upload instead',  width=20, command=lambda:raiseFrame(Upload))
    open_button.grid(row=1, column=0)
    
    lastx, lasty = None, None
    image_number = 0
    global image_can, draw
    def upload_file(Upload, model):
        global img
        f_types = [('JPG Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        
        img = Image.open(filename)
        img_resized = img.resize((200,200))
        img = ImageTk.PhotoImage(img_resized)
        image = ttk.Button(Upload, image=img) 
        image.grid(row=3, column=0)

        value = image_processing(model, filename)
        output = "Predicted numerical digit: " + str(value)
        text2 = tk.Label(Upload, text=output)
        text2.grid(row=4, column=0)

    def save(Create, model, image_number):
        global image_can
        filename = f'image_{image_number}.jpg'  
        image_can.save(filename)
        image_number += 1
        
        image = Image.open(filename).convert('L')
        image = image.resize((28,28))
        img = 255-np.array(image)

        prediction = model.predict(img.reshape(1,784))  
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)
        
        output = "Predicted numerical digit: " + str(yhat)
        text2 = tk.Label(Create, text=output)
        text2.grid(row=5, column=0)

    def clean(Create, paint_canvas):
        global image_can, draw
        image_can = PIL.Image.new('RGB', (200, 200), 'white')
        paint_canvas.delete('all')
        draw = ImageDraw.Draw(image_can)
   
    def activate_paint(e):
        global lastx, lasty
        paint_canvas.bind('<B1-Motion>', paint)
        lastx, lasty = e.x, e.y

    def paint(e):
        global lastx, lasty
        x, y = e.x, e.y
        paint_canvas.create_line((lastx, lasty, x, y), width=10)
        draw.line((lastx, lasty, x, y), fill='black', width=10)
        lastx, lasty = x, y

    paint_canvas = Canvas(Create, width=200, height=200, bg='white')
    image_can = PIL.Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(image_can)

    paint_canvas.bind('<1>', activate_paint)
    paint_canvas.grid(column=0, row=2)

    clear_button = ttk.Button(Create, text='Clear Canvas',  width=20, command=lambda:clean(Create, paint_canvas))
    clear_button.grid(column=0, row=3)
    
    predict_button = ttk.Button(Create, text="Save and Predict", command=lambda:save(Create, model, image_number))
    predict_button.grid(column=0, row=4)
    
    return Upload, Create

def raiseFrame(frame):
    frame.tkraise()
    
createWindow()