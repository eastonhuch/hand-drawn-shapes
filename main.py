#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 10:26:54 2017
License: MIT
@author: easton
"""

#%cd ~/Documents/Shape Recognition/Drawing App/Draw Shapes/

import drive
import draw

def main():
	draw.what_to_draw()
	draw.draw()
	drive.upload(draw.get_files())
	#drive.view_files()

if __name__ == "__main__":
	main()
    
