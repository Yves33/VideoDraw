import numpy as np
import cv2

DEFAULT_ROTATION=1
DEFAULT_ZOOM=1.0
helptext='''Stream your drawings.
"=" : resets all transformation
"left click" : selects 4 points for geometric transform
"+" : zoom in
"-" : zoom out
" " : rotates
"q" : exits
"s" : saves picture
"h" : toggle help message
"a" : autodetect (not implemented yet)
'''
helppos:50
helpstyle={
    'fontFace':cv2.FONT_HERSHEY_SIMPLEX,
    'fontScale':0.8,
    'color':(255,0,0),
    'thickness':2
    }

## end of parameter section
rotation=DEFAULT_ROTATION
zoom=DEFAULT_ZOOM
contrast=1.00
brightness=50
help=False
points=[]

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

def correct_image(img, contrast=1.25, brightness=50):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:,:,2] = np.clip(contrast * img[:,:,2] + brightness, 0, 255)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def mousecallback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points)>4:
            points=[]

def oncontrast(x):
    global contrast
    contrast=x/30

def onbrightness(x):
    global brightness
    brightness=x

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
ret_val, img = cam.read()
cv2.imshow('Live draw', img)
cv2.createTrackbar('contrast', 'Live draw', 0, 300, oncontrast)
cv2.createTrackbar('brightness', 'Live draw', 0, 100, onbrightness)
cv2.setMouseCallback('Live draw', mousecallback)
while True:
    ret_val, img = cam.read()
    if len(points)==4:
        img=four_point_transform(img,order_points(np.array(points)))
        for r in range(rotation):
            img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
            img=cv2.resize(img, None, fx=zoom, fy=zoom)
            ##img=cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
            img=correct_image(img,contrast, brightness)
    else:
        for p in points:
            cv2.circle(img,p,5,(0,255,0),-1)
    if help:
        for e,l in enumerate(helptext.split('\n')):
            cv2.putText(img,l,(50,50+e*20),**helpstyle)
    cv2.imshow('Live draw', img)
    ## event handling starts here
    key=cv2.waitKey(1)
    if key==ord(' '):
        rotation+=1
        rotation=rotation%4
    elif key==ord('+'):
        zoom*=1.05
    elif key==ord('-'):
        zoom/=1.05
    elif key==ord('='):
        zoom=DEFAULT_ZOOM
        rotation=DEFAULT_ROTATION
        points=[]
    elif key==ord('h'):
        help=not help
    elif key==27 or key==ord('q'): 
        break  # esc to quit
cv2.destroyAllWindows()