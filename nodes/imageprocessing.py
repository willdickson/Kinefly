#!/usr/bin/env python
import rospy
import copy
import cv2
import numpy as np


class TransformException(Exception):
    pass


def get_angle_from_points_i(pt1, pt2):
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    return np.arctan2(y,x)


# Intersection of two lines, given two points on each line.
def get_intersection(pt1a, pt1b, pt2a, pt2b):
    # Line 1
    x1 = pt1a[0]
    y1 = pt1a[1]
    x2 = pt1b[0]
    y2 = pt1b[1]
    
    # Line 2
    x3 = pt2a[0]
    y3 = pt2a[1]
    x4 = pt2b[0]
    y4 = pt2b[1]
    
    den = (x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)
    if (den != 0.0):
        x = (x1*x3*y2 - x2*x3*y1 - x1*x4*y2 + x2*x4*y1 - x1*x3*y4 + x1*x4*y3 + x2*x3*y4 - x2*x4*y3) / den
        y = (x1*y2*y3 - x2*y1*y3 - x1*y2*y4 + x2*y1*y4 - x3*y1*y4 + x4*y1*y3 + x3*y2*y4 - x4*y2*y3) / den
    else:
        x = x3   
        y = y3   
    
    return np.array([x,y])
            
            
# get_projection_onto_axis()
# Project the given point onto the axis.
#
def get_projection_onto_axis(ptAnywhere, (ptAxisBase, ptAxisHead)):
    # Project the point onto the body axis.
    ptB = ptAxisHead - ptAxisBase
    ptM = ptAnywhere - ptAxisBase
    ptAxis = np.dot(ptB,ptM) / np.dot(ptB,ptB) * ptB + ptAxisBase
        
    return ptAxis
    
            
def get_reflection_across_axis(ptAnywhere, (ptAxisBase, ptAxisHead)):
    ptAxis = get_projection_onto_axis(ptAnywhere, (ptAxisBase, ptAxisHead))
    ptReflected = ptAnywhere + 2*(ptAxis-ptAnywhere)
    
    return ptReflected

    
def filter_median(data, q=1): # q is the 'radius' of the filter window.  q==1 is a window of 3.  q==2 is a window of 5.
    data2 = copy.copy(data)
    for i in range(q,len(data)-q):
        data2[i] = np.median(data[i-q:i+q+1]) # Median filter of window.

    # Left-fill the first values.
    try:
        data2[0:q] = data2[q]
    
        # Right-fill the last values.
        data2[len(data2)-q:len(data2)] = data2[-(q+1)]
        dataOut = data2
        
    except IndexError:
        dataOut = data
        
    return dataOut
        

def clip(x, lo, hi):
    return max(min(x,hi),lo)
    
    
# clip a point to the image shape.  pt is (x,y), and shape is (yMax+1, xMax+1)
def clip_pt(pt, shape):
    return (clip(pt[0], 0, shape[1]-1), clip(pt[1], 0, shape[0]-1))
    
    
    

###############################################################################
###############################################################################
class PolarTransforms(object):
    def __init__(self):
        self._transforms = {}

        
    def _get_transform_polar_log(self, i_0, j_0, i_n, j_n, nRho, dRho, nTheta, theta_0, theta_1):
        transform = self._transforms.get((i_0, j_0, i_n, j_n, nRho, nTheta, theta_0, theta_1))
    
        if (transform == None):
            i_k     = []
            j_k     = []
            rho_k   = []
            theta_k = []
    
            aspect = float(i_n) / float(j_n)
            dTheta = (theta_1 - theta_0) / nTheta
            for iRho in range(0, nRho):
                rho = np.exp(iRho * dRho)
                
                for iTheta in range(0, nTheta):
                    theta = theta_0 + iTheta * dTheta
    
                    # i,j points on a circle.
                    i_c = rho * np.sin(theta)
                    j_c = rho * np.cos(theta)
                    
                    # Expand the circle onto an ellipse in the larger dimension.
                    if (aspect>=1.0):
                        i = i_0 + int(i_c * aspect)
                        j = j_0 + int(j_c)
                    else:
                        i = i_0 + int(i_c)
                        j = j_0 + int(j_c / aspect)
    
                    if (0 <= i < i_n) and (0 <= j < j_n):
                        i_k.append(i)
                        j_k.append(j)
                        rho_k.append(iRho)
                        theta_k.append(iTheta)
    
            transform = ((np.array(rho_k), np.array(theta_k)), (np.array(i_k), np.array(j_k)))
            self._transforms[i_0, j_0, i_n, j_n, nRho, nTheta, theta_0, theta_1] = transform
    
        return transform
    
    # transform_polar_log()
    # Remap an image into log-polar coordinates, where (i_0,j_0) is the (y,x) origin in the original image.
    # nRho:         Number of radial (vert) pixels in the output image.
    # amplifyRho:   Multiplier of vertical output dimension instead of nRho.
    # nTheta:       Number of angular (horiz) pixels in the output image.
    # amplifyTheta: Multiplier of horizontal output dimension instead of nTheta.
    # theta_0:      Starting angle.
    # theta_1:      Ending angle.
    # scale:        0.0=Include to the nearest side (all pixels in output image are valid); 1.0=Include to the 
    #               farthest corner (some pixels in output image from outside the input image). 
    #
    # Credit to http://machineawakening.blogspot.com/2012/02
    #
    def transform_polar_log(self, image, i_0, j_0, nRho=None, amplifyRho=1.0, nTheta=None, amplifyTheta=1.0, theta_0=0.0, theta_1=2.0*np.pi, scale=0.0):
        (i_n, j_n) = image.shape[:2]
        
        i_c = max(i_0, i_n - i_0)
        j_c = max(j_0, j_n - j_0)
        d_c = (i_c ** 2 + j_c ** 2) ** 0.5 # Distance to the farthest image corner.
        d_s = min(i_0, i_n-i_0, j_0, j_n-j_0)  # Distance to the nearest image side.
        d = scale*d_c + (1.0-scale)*d_s
        
        if (nRho == None):
            nRho = int(np.ceil(d*amplifyRho))
        
        if (nTheta == None):
            #nTheta = int(np.ceil(j_n * amplifyTheta))
            nTheta = int(amplifyTheta * 2*np.pi*np.sqrt((i_n**2 + j_n**2)/2)) # Approximate circumference of ellipse in the roi.
        
        dRho = np.log(d) / nRho
        
        (pt, ij) = self._get_transform_polar_log(i_0, j_0, 
                                           i_n, j_n, 
                                           nRho, dRho, 
                                           nTheta, theta_0, theta_1)
        imgTransformed = np.zeros((nRho, nTheta) + image.shape[2:], dtype=image.dtype)
        imgTransformed[pt] = image[ij]

        return imgTransformed


    def _get_transform_polar_elliptical(self, i_0, j_0, i_n, j_n, (raxial, rortho), drStrip, angleEllipse, nRho, nTheta, theta_0, theta_1, rClip):
        nTheta = max(1,nTheta)
        transform = self._transforms.get((i_0, j_0, i_n, j_n, nRho, drStrip, nTheta, theta_0, theta_1, rClip))
    
        if (transform == None):
            i_k     = []
            j_k     = []
            rho_k   = []
            theta_k = []

#             # Convert from circular angles to elliptical angles.
#             xy_e = np.array([raxial*np.cos(theta_0), rortho*np.sin(theta_0)])
#             theta_0e = np.arctan2(xy_e[1], xy_e[0])
#             xy_e = np.array([raxial*np.cos(theta_1), rortho*np.sin(theta_1)])
#             theta_1e = np.arctan2(xy_e[1], xy_e[0])

            # Radii of the inner and outer ellipses.
            raxial_outer = raxial + drStrip
            rortho_outer = raxial + drStrip
            raxial_inner = raxial - drStrip
            rortho_inner = raxial - drStrip
            
            R = np.array([[np.cos(-angleEllipse), -np.sin(-angleEllipse)], 
                          [np.sin(-angleEllipse), np.cos(-angleEllipse)]])    
            dTheta = (theta_1 - theta_0) / nTheta
            
            # Step through the wedge from theta_0 to theta_1.
            for iTheta in range(0, nTheta):
                theta = theta_0 + iTheta * dTheta

                # Convert from circular angles to elliptical angles.
                xy_e = np.array([raxial*np.cos(theta), rortho*np.sin(theta)])
                theta_e = np.arctan2(xy_e[1], xy_e[0])

                # Radii of ellipses at this angle.
                #rho_e = np.linalg.norm([2*raxial*np.cos(theta), 2*rortho*np.sin(theta)])
                rho_e_inner = np.linalg.norm([raxial_inner*np.cos(theta), rortho_inner*np.sin(theta)])
                rho_e_outer = np.linalg.norm([raxial_outer*np.cos(theta), rortho_outer*np.sin(theta)])
                dRho = (rho_e_outer - rho_e_inner) / nRho
                
                # Step along the radius from the inner to the outer ellipse.  rClip is a clip factor on range [0,1]
                for iRho in range(0, int(np.ceil(rClip*nRho))):
                    rho = rho_e_inner + iRho * dRho

                    # i,j points on an upright ellipse.
                    i_e = rho * np.sin(theta_e)
                    j_e = rho * np.cos(theta_e)
                    
                    # Rotate the point.
                    ij = R.dot([i_e, j_e])
                    
                    # Translate it into position.
                    i = int(i_0 + ij[0])
                    j = int(j_0 + ij[1])
                    
    
                    # Put the transform values into the lists.
                    if (0 <= i < i_n) and (0 <= j < j_n):
                        i_k.append(i)
                        j_k.append(j)
                        rho_k.append(iRho)
                        theta_k.append(iTheta)
    
            transform = ((np.array(rho_k), np.array(theta_k)), (np.array(i_k), np.array(j_k)))
            self._transforms[i_0, j_0, i_n, j_n, nRho, drStrip, nTheta, theta_0, theta_1, rClip] = transform
    
        return transform
    
    # transform_polar_elliptical()
    # Remap an image into linear-polar coordinates, where (i_0,j_0) is the (y,x) origin in the original image.
    # raxial:       Ellipse radius at angleEllipse.
    # rortho:       Ellipse radius orthogonal to angleEllipse.
    # dradiusStrip: Half-width of the mask strip in the axial direction.
    # nRho:         Number of radial (vert) pixels in the output image.
    # amplifyRho:   Multiplier of vertical output dimension instead of nRho.
    # rClip:        Fraction of the the image that should be calculated vertically.  i.e. Limit r when calculating pixels.
    # angleEllipse: Axis of ellipse rotation.
    # theta_0:      Circular angle to one side of ellipse angle.
    # theta_1:      Circular angle to other side of ellipse angle.
    # nTheta:       Number of angular (horiz) pixels in the output image.
    # amplifyTheta: Multiplier of horizontal output dimension instead of nTheta.
    #
    #
    def transform_polar_elliptical(self, image, i_0, j_0, raxial=None, rortho=None, dradiusStrip=None, nRho=None, amplifyRho=1.0, rClip=0.0, angleEllipse=0.0, theta_0=-np.pi, theta_1=np.pi, nTheta=None, amplifyTheta=1.0):
        (i_n, j_n) = image.shape[:2]
        if (raxial is None):
            raxial = i_n / 2
        if (rortho is None):
            rortho = j_n / 2
            
        if (dradiusStrip is None):
            dradiusStrip = raxial-5
        
#         i_c = max(i_0, i_n-i_0)
#         j_c = max(j_0, j_n-j_0)
#         
#         d_c = (i_c ** 2 + j_c ** 2) ** 0.5            # Distance to the farthest image corner.
#         d_s_near = min(i_0, i_n-i_0, j_0, j_n-j_0)    # Distance to the nearest image side.
#         d_s_far = max(i_0, i_n-i_0, j_0, j_n-j_0)     # Distance to the farthest image side.

        
        # Radii of the inner and outer ellipses.
        raxial_outer = raxial + dradiusStrip
        rortho_outer = raxial + dradiusStrip
        raxial_inner = raxial - dradiusStrip
        rortho_inner = raxial - dradiusStrip

            
        # Nearest nonzero distance of point (i_0,j_0) to a side.
        #d_sides = [i_0, i_n-i_0, j_0, j_n-j_0]
        #d_nonzero = d_sides[np.where(d_sides>0)[0]]
        #d_s_0 = d_nonzero.min()
        

        # Distance to nearest point of outer elliptical wedge of point (i_0,j_0).
        d_e_raxial_outer = raxial_outer # Distance to the axial point.
        d_e_rortho_outer = rortho_outer # Distance to the ortho point.
        d_e_wedge_outer = np.linalg.norm([raxial_outer*np.cos(theta_0), rortho_outer*np.sin(theta_0)]) # Distance to the theta_0 wedge point.
        if (np.abs(theta_0) >= np.pi/2.0):
            d_e_min_outer = min(d_e_raxial_outer, d_e_wedge_outer, d_e_rortho_outer) # Nearest of the three.
        else:
            d_e_min_outer = min(d_e_raxial_outer, d_e_wedge_outer) # Nearest of the three.

        # Distance to nearest point of inner elliptical wedge of point (i_0,j_0).
        d_e_raxial_inner = raxial_inner # Distance to the axial point.
        d_e_rortho_inner = rortho_inner # Distance to the ortho point.
        d_e_wedge_inner = np.linalg.norm([raxial_inner*np.cos(theta_0), rortho_inner*np.sin(theta_0)]) # Distance to the theta_0 wedge point.
        if (np.abs(theta_0) >= np.pi/2.0):
            d_e_min_inner = min(d_e_raxial_inner, d_e_wedge_inner, d_e_rortho_inner) # Nearest of the three.
        else:
            d_e_min_inner = min(d_e_raxial_inner, d_e_wedge_inner) # Nearest of the three.

        
        d = d_e_min_outer - d_e_min_inner 
        
        
        # Convert from circular angles to elliptical angles.
        xy_e = np.array([raxial*np.cos(theta_0), rortho*np.sin(theta_0)])
        theta_0e = np.arctan2(xy_e[1], xy_e[0])
        xy_e = np.array([raxial*np.cos(theta_1), rortho*np.sin(theta_1)])
        theta_1e = np.arctan2(xy_e[1], xy_e[0])
        
        
        if (nRho == None):
            nRho = int(np.ceil(d * amplifyRho))
        
        # Number of angular steps depends on the number of pixels along the elliptical arc, ideally 1 step per pixel.
        if (nTheta == None):
            circumference = 2*np.pi*np.sqrt(((raxial)**2 + (rortho)**2)) # Approximate circumference of ellipse.
            fraction_wedge = np.abs(theta_1e - theta_0e)/(2*np.pi) # Approximate fraction that is the wedge of interest.
            nTheta = int(amplifyTheta * circumference * fraction_wedge)  
        
        (pt, ij) = self._get_transform_polar_elliptical(i_0, j_0, 
                                           i_n, j_n,
                                           (raxial, rortho),
                                           dradiusStrip,
                                           angleEllipse, 
                                           nRho, 
                                           nTheta, theta_0, theta_1, rClip)
        imgTransformed = np.zeros((nRho, nTheta) + image.shape[2:], dtype=image.dtype)
        if (len(pt[0])>0):
            imgTransformed[pt] = image[ij]
        else:
            # No points transformed.
            raise TransformException()
        
        return imgTransformed

# End class PolarTransforms


###############################################################################
###############################################################################
class PhaseCorrelation(object):
    # get_shift()
    # Calculate the coordinate shift between two images.
    #
    def get_shift(self, imgA, imgB):
        rv = np.array([0.0, 0.0])
        if (imgA is not None) and (imgB is not None) and (imgA.shape==imgB.shape):
            # Phase correlation.
            A  = cv2.dft(imgA)
            B  = cv2.dft(imgB)
            AB = cv2.mulSpectrums(A, B, flags=0, conjB=True)
            normAB = cv2.norm(AB)
            if (normAB != 0.0):
                crosspower = AB / normAB
                shift = cv2.idft(crosspower)
                shift0  = np.roll(shift,  int(shift.shape[0]/2), 0)
                shift00 = np.roll(shift0, int(shift.shape[1]/2), 1) # Roll the matrix so 0,0 goes to the center of the image.
                
                # Get the coordinates of the maximum shift.
                kShift = np.argmax(shift00)
                (iShift,jShift) = np.unravel_index(kShift, shift00.shape)
    
                # Get weighted centroid of a region around the peak, for sub-pixel accuracy.
                w = 7
                r = int((w-1)/2)
                i0 = clip(iShift-r, 0, shift00.shape[0]-1)
                i1 = clip(iShift+r, 0, shift00.shape[0]-1)+1
                j0 = clip(jShift-r, 0, shift00.shape[1]-1)
                j1 = clip(jShift+r, 0, shift00.shape[1]-1)+1
                peak = shift00[i0:i1].T[j0:j1].T
                moments = cv2.moments(peak, binaryImage=False)
                           
                if (moments['m00'] != 0.0):
                    iShiftSubpixel = moments['m01']/moments['m00'] + float(i0)
                    jShiftSubpixel = moments['m10']/moments['m00'] + float(j0)
                else:
                    iShiftSubpixel = float(shift.shape[0])/2.0
                    jShiftSubpixel = float(shift.shape[1])/2.0
                
                # Accomodate the matrix roll we did above.
                iShiftSubpixel -= float(shift.shape[0])/2.0
                jShiftSubpixel -= float(shift.shape[1])/2.0
    
                # Convert unsigned shifts to signed shifts. 
                height = float(shift00.shape[0])
                width  = float(shift00.shape[1])
                iShiftSubpixel  = ((iShiftSubpixel+height/2.0) % height) - height/2.0
                jShiftSubpixel  = ((jShiftSubpixel+width/2.0) % width) - width/2.0
                
                rv = np.array([iShiftSubpixel, jShiftSubpixel])

            
        return rv
        
        
###############################################################################
###############################################################################
class WindowFunctions(object):
    # create_hanning()
    # Create a 2D Hanning window function.
    #
    def create_hanning(self, shape):
        (height,width) = shape
        wfn = np.ones(shape, dtype=np.float32)
        if (height>1) and (width>1):
            for i in range(width):
                 for j in range(height):
                     x = 2*np.pi*i/(width-1) # x ranges 0 to 2pi across the image width
                     y = 2*np.pi*j/(height-1) # y ranges 0 to 2pi across the image height
                     wfn[j][i] = 0.5*(1-np.cos(x)) * 0.5*(1-np.cos(y))
                 
        return wfn


    # create_tukey()
    # Create a 2D Tukey window function.
    #
    # alpha:    # Width of the flat top.  alpha==0 gives rectangular, alpha=1 gives Hann.
    #
    def create_tukey(self, shape, alpha=0.25):
        (height,width) = shape
        wfn = np.ones(shape, dtype=np.float32)
        if (height>1) and (width>1):
            for i in range(width):
                for j in range(height):
                    #y = np.pi*(2*j/(alpha*(height-1))-1)
                    
                    if (alpha<=0.0) or (alpha>1.0):
                        x = 0.0
                        y = 0.0
                    else:
                        if (0 <= i <= (alpha*(width-1))/2):
                            x = np.pi*(2*i/(alpha*(width-1))-1)
                        elif ((alpha*(width-1))/2 < i <= (width-1)*(1-alpha/2)):
                            x = 0.0
                        elif ((width-1)*(1-alpha/2) < i <= width-1):
                            x = np.pi*(2*i/(alpha*(width-1))-2/alpha+1)
                            
                        if (0 <= j <= (alpha*(height-1))/2):
                            y = np.pi*(2*j/(alpha*(height-1)) - 1)
                        elif ((alpha*(height-1))/2 < j <= (height-1)*(1-alpha/2)):
                            y = 0.0
                        elif ((height-1)*(1-alpha/2) < j <= height-1):
                            y = np.pi*(2*j/(alpha*(height-1)) - 2/alpha + 1)
                    
                    wfnx = 0.5*(1+np.cos(x))
                    wfny = 0.5*(1+np.cos(y))
                    wfn[j][i] = wfnx * wfny
                 
        return wfn

# End class WindowFunctions


    

