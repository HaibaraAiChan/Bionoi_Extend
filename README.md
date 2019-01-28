# Bionoi_Extend
The original version bionoi only has XOY principle direction projection.  

The extend version includes 6 projection directions, 4 rotation angles and 3 flip directions, 72 images in total.  

Projection direction:  

         XOY+, XOY-, YOZ+, YOZ-, ZOX+, ZOX-  
	
Rotate angle:  

          0, 90, 180, 270  
	 
Flip  direction:  

          original, up-down, left-right  
#Examples  

./voronoi.py
or
./voronoi.py -mol 4v94E.mol2 -out ./output/ -dpi 120 -alpha 0.5 -size 128 -proDirect 1 -rotAngle2D 0 -flip 0
