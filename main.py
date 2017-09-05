#!/usr/bin/env python3
# -*- coding: utf-8 -*-
%cd ~/Documents/Shape Recognition/Drawing App/Draw Shapes/

import drive
import draw

def main():
	draw.what_to_draw()
	draw.draw()
	drive.upload(draw.get_files())
	#drive.view_files()

if __name__ == "__main__":
	main()
    
