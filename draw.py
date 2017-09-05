#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:55:45 2017
License: MIT
@author: easton
"""

import tkinter
import pandas as pd
import datetime

b1 = "up"
points, files = [], []
xold, yold = None, None
num_shapes, shape_number = 0, 0
shapes = ['Block_Arrow', 'Parallelagram','Curved_Line','Cross','Heart',
          'Diamond', 'Pentagon','Acute_Isoceles_Triangle','2-Way_Block_Arrow',
          'Circle', 'Square', 'Straight_Line', 'Bent_Line', 'Octagon', 
          'Callout', 'Star', 'Hexagon', 'Right_Triangle','Cloud']
shapes_per_category = 1
name = 'unknown'

def get_files():
	return files

def what_to_draw():
    global shapes, shape_number, shapes_per_category, v, name
    name = input('what\'s your name? ')
    print('These are the shapes you can draw:')
    for shape in shapes:
        print(shape)
    user_input = input('Do you want to draw these (yes/no)? ')
    if user_input.lower() == 'no':
        shapes = [input('What shape do you want to draw (enter spaces as underscores)? ')]
    shapes_per_category = int(input('How many drawings do you want to make of each shape? '))
    v.set('draw a ' + shapes[shape_number] + ' 1/' + str(shapes_per_category))

def mouse_down(event):
    global b1
    b1 = "down"

def mouse_up(event):
    global b1, xold, yold, points, drawing_area, num_shapes, shapes, shape_number, shapes_per_category, files, v
    num_shapes += 1
    drawing_area.delete("all")
    points_df = pd.DataFrame(points, columns=['x','y'])
    timestamp = datetime.datetime.now().strftime(" %m-%d-%Y %H%M%S%f")
    filename = './Shapes/' + shapes[shape_number] + ' ' + name + timestamp + '.csv'
    files.append(filename)
    points_df.to_csv(filename, index = False)
    points = []
    b1 = "up"
    xold = None
    yold = None
    v.set('draw a ' + shapes[shape_number] + ' ' + str(num_shapes) + '/' + str(shapes_per_category))


def motion(event):
    if b1 == "down":
        global xold, yold, points
        points.append([event.x, event.y])
        if xold is not None and yold is not None:
            event.widget.create_line(xold,yold,event.x,event.y,smooth=tkinter.TRUE)
        xold = event.x
        yold = event.y

root = tkinter.Tk()
root.title("Draw a Shape")
root.geometry("500x500")
drawing_area = tkinter.Canvas(root)
v = tkinter.StringVar()
tkinter.Label(root, textvariable=v).pack()
drawing_area.pack(expand=tkinter.YES, fill=tkinter.BOTH)
drawing_area.bind("<Motion>", motion)
drawing_area.bind("<ButtonPress-1>", mouse_down)
drawing_area.bind("<ButtonRelease-1>", mouse_up)
		
def draw():		
    global shapes, shape_number, num_shapes, shapes_per_category, root, v   
    while True:
    	    if num_shapes >= shapes_per_category:
    	        num_shapes = 0
    	        shape_number += 1
    	        if shape_number >= len(shapes):
    	            print('you\'re done!')
    	            root.destroy()
    	            break	       
    	    root.update_idletasks()
    	    root.update()