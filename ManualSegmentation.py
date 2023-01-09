import matplotlib
from matplotlib import image
from matplotlib.widgets import Button
from skimage import io, feature, filters, morphology
from skimage.color import rgb2gray
from skimage.morphology import flood_fill
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from random import randint
from scipy.optimize import curve_fit

#PARAMETERS:
CANNY_sigma = 3
CANNY_lowt = 0.04 
CANNY_hight = 0.12   
HYST_lowt_vals = [0.35,0.35,0.35,0.5]
HYST_hight_vals = [0.63,0.68,0.65,0.75]
cutoff_radius = 15
side = 'dorsal'
muscle = 'FPB'


class Scan:
    def __init__(self,folder,start,stop,seg_range):
        self.seg_range=[x-start for x in seg_range]
        self.muscles = muscles
        self.start = start
        self.stop = stop
        self.folder = folder
        self.scan_folder = subject_path + folder
        self.files = os.listdir(self.scan_folder)
        self.images=[]
        for i in range(self.start,self.stop):
            filename = self.scan_folder + '//' + self.files[i]
            img = rgb2gray(io.imread(filename))[60:550,528:1480]
            TRI = TRI_all[int(self.folder),i]
            current_img = self.usImage(img,TRI,self.images,int(self.folder))
            self.images.append(current_img)

    def skin_reconstruct(self):
        for img in self.images:
            img.get_skin_surf()
            try:
                self.skin_points = np.vstack([self.skin_points,img.skin_pts_RCS])
            except:
                self.skin_points = np.array(img.skin_pts_RCS)
        self.get_max_skin_point()

    def segment_images(self):
        for img in self.images[self.seg_range[0]:self.seg_range[1]]:
            current_img=img
            current_img.skip = False
            self.muscles = current_img.segment(self.shift,self.muscles,self.seg_range)
            if current_img.skip == True:
                continue 

            if current_img.skip_d == False: 
                try:
                    self.dorsal_points = np.vstack([self.dorsal_points,current_img.dorsal_pts_RCS]) 
                except:
                    self.dorsal_points = np.array(current_img.dorsal_pts_RCS)

    def get_max_skin_point(self):
        point_data = self.skin_points
        def function(data, a0, a1,a2, a3, a4, a5):
            x = data[0]
            y = data[1]
        
            return a0+a1*x+a2*y+a3*x*y+a4*x**2+a5*y**2

        x_data = []
        y_data = []
        z_data = []
        for item in point_data:
            x_data.append(item[0])
            y_data.append(item[1])
            z_data.append(item[2])
        # get fit parameters from scipy curve fit
        parameters, covariance = curve_fit(function, [x_data, y_data], z_data)

        # create surface function model
        # setup data points for calculating surface model
        model_x_data = np.linspace(min(x_data)-10, max(x_data)+10, 30)
        model_y_data = np.linspace(min(y_data)-10, max(y_data)+10, 30)
        # create coordinate arrays for vectorized evaluations
        X, Y = np.meshgrid(model_x_data, model_y_data)

        # calculate Z coordinate array
        Z = function(np.array([X, Y]), *parameters)
        x,y,z =X.flatten(), Y.flatten(), Z.flatten()
        points = np.stack([x,y,z],axis=1)
        self.max_skin_point = points[np.where(points[:,2] == np.max(points[:,2]))]

    def shift_points(self,other_msp):
        shift = other_msp-self.max_skin_point
        self.dorsal_points += shift
        self.skin_points += shift 

    class usImage:
        def __init__(self,img,TRI,images,scan_n):
            self.img = img
            self.TRI = TRI
            self.images = images
            self.n=scan_n
            self.hyst_lowt = HYST_lowt_vals[scan_n]
            self.hyst_hight = HYST_hight_vals[scan_n]
            self.current_img_index=len(images)

        def segment(self,shift,muscles,seg_range):
            self.muscles = muscles
            self.shift=shift
            print("scan: ", self.n, "image: ",self.current_img_index," out of ",seg_range[1])
            self.project_muscle()

            self.redo,first,self.skip_d,self.skip_v = False,True,False,False
            while self.redo == True or first == True:
                self.skip = False
                self.get_dorsal_pts()
                if self.skip == True:
                    return self.muscles
                first = False
                if self.redo == False:
                    self.show_plot()
                if self.skip_d == True or self.skip_v == True:
                    break
            self.dorsal_pts_RCS = self.ICS_to_RCS(self.dorsal_pts)
            return self.muscles

        def project_muscle(self):
            TRI = self.TRI
            TIR = np.linalg.inv(TRI)
            shifted_current = current_muscle + self.shift
            for muscle in self.muscles:
                ind_del = []
                i=0
                shifted_muscle = muscle + self.shift
                for point in shifted_muscle:
                    point = np.append(point,[1])
                    point_ICS = np.delete(np.matmul(TIR,point),3)
                    if point_ICS[2]>0.1:
                        ind_del.append(i)
                    if abs(point_ICS[2])<0.1: 
                        point_ICS = np.delete(point_ICS,2)/0.066666666667                     
                        try:
                            self.pts_ICS = np.vstack([self.pts_ICS,point_ICS])
                        except:
                            self.pts_ICS = np.array(point_ICS,ndmin=2)
                    i += 1
                if len(ind_del)>0:
                    muscle = np.delete(muscle,ind_del,axis=0)
                print('points left: ',muscle.shape[0])

                try:
                    self.pts_ICS
                except:
                    self.pts_ICS = np.array([0,0],ndmin=2)
                
            for point in shifted_current:
                point = np.append(point,[1])
                point_ICS = np.delete(np.matmul(TIR,point),3)
                if abs(point_ICS[2])<0.1: 
                    point_ICS = np.delete(point_ICS,2)/0.0666666667
                    try:
                        self.current_pts_ICS = np.vstack([self.current_pts_ICS,point_ICS])
                    except:
                        self.current_pts_ICS = np.array(point_ICS,ndmin=2)    

            try:
                self.current_pts_ICS
            except:
                self.current_pts_ICS = np.array([0,0],ndmin=2)


        def get_skin_surf(self):
            hyst = self.apply_hyst_thresh()
            footprint = morphology.disk(2)
            res_bright = morphology.white_tophat(hyst, footprint) 
            indices_bright = np.where(res_bright == True)
            indices_bright = np.stack([indices_bright[0],indices_bright[1]],axis=1)
            for i in indices_bright:
                hyst[i[0],i[1]] = False

            i=0
            hyst_T = np.transpose(hyst)
            for col in hyst_T:
                j=0
                bright_found=False
                for pixel in col:
                    pixel_coord = [i,j]
                    if j<40:
                        j+=1
                        continue
                    if pixel == True and bright_found == False:
                        try:
                            #only add to skin_surf if j is within 10 pixels of last j (skips bright particles/noise above skin surface)
                            if abs(self.skin_surf[-1,1]-j)<10:
                                self.skin_surf = np.vstack([self.skin_surf,pixel_coord])
                                bright_found=True
                        except:
                            self.skin_surf = np.array(pixel_coord,ndmin=2)
                            bright_found=True
                    j+=1
                i+=1

            #get max skin surface point
            skin_peak_y = np.min(self.skin_surf[:,1])
            skin_peak_x = self.skin_surf[np.where(self.skin_surf[:,1]==skin_peak_y)[0][0]]    
            self.skin_peak = np.append(skin_peak_x,skin_peak_y)

            self.skin_pts_RCS = self.ICS_to_RCS(self.skin_surf)
        
        def create_line(self,points):
            points = points
            line_pts = np.array(points[0],ndmin=2)
            for i in range(0,len(points)-1):
                f=np.polyfit(points[i:i+2,0],points[i:i+2,1],deg=1)
                x=np.arange(points[i,0],points[i+1,0],0.2)
                y = np.polyval(f,x)
                line_pts=np.vstack([line_pts,np.stack([x,y],axis=-1)])
            return line_pts

        
        def get_dorsal_pts(self):
            #set img to cropped img and adjust volar points to fit cropped image
            #APB_points = self.pts_ICS - [self.x_adj,self.y_adj]
            img = self.img
              
            #get points near muscle boundary, if first image: manual selection; otherwise, use previous b2mpeak
            #if manual selection, cutoff is set to volar points. If using previous b2mpeak, cutoff is set to previously selected points
            clicked_points = click_point_coordinate(img*255)
            if clicked_points[0,0]<0:
                self.skip=True
                return 
            ind = np.lexsort((clicked_points[:,1],clicked_points[:,0]))
            clicked_points = clicked_points[ind]
            self.dorsal_pts = self.create_line(clicked_points)
            #print("clicked points: ",self.dorsal_pts)


            #Adjust extracted dorsal points to Image Coordinate System
            #self.dorsal_pts[:,0] += self.x_adj
            #self.dorsal_pts[:,1] += self.y_adj

            self.redo = False

        def crop_image(self):
            ymin, ymax = self.skin_peak[1], self.skin_peak[1]+350 
            self.cropped_img = self.img[ymin:ymax,:]
            
            sigma=5
            #blurred = skimage.filters.gaussian(self.cropped_img, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
            blurred = filters.apply_hysteresis_threshold(self.cropped_img,0.4,0.75)
            blurred = binary_to_gs(blurred)
            """plt.imshow(blurred)
            plt.show()"""
            min_point = np.where(blurred[:,self.skin_peak[0]]==np.min(blurred[5:,self.skin_peak[0]]))[0][0]
            blurred=blurred*255
            flood = flood_fill(blurred,(min_point,self.skin_peak[0]),255,tolerance=100-blurred[min_point,self.skin_peak[0]])
            skin_flood = flood_fill(blurred,(self.skin_peak[1]-1,self.skin_peak[0]+2),255,tolerance=35)

            muscle_coords = np.where(flood == 255)
            skin_coords = np.where(skin_flood == 255)
            muscle_tip = np.min(muscle_coords[1])

            xmin, xmax = muscle_tip-10, np.max(self.skin_surf[:,0]-30)
            if self.n == 2 or self.n==1:
                xmax = np.max(self.skin_surf[:,0])
            if self.n == 0 or self.n ==1:
                xmin = 0
            if self.n == 0:
                xmax+=30
            if self.n == 2 and self.current_img_index < 10 and muscle == 'APB':
                xmin = 230
            if self.n == 3: 
                xmax -= 10
            if xmin<0:
                xmin=0            

            #delete points from muscle and skin surf that are cropped out
            if muscle == 'APB':
                self.volar_pts = np.delete(self.volar_pts,np.where(self.volar_pts[:,1]>ymax),axis=0)
                self.volar_pts = np.delete(self.volar_pts,np.where(self.volar_pts[:,0]<xmin),axis=0)
                self.volar_pts = np.delete(self.volar_pts,np.where(self.volar_pts[:,0]>xmax),axis=0)
            self.skin_surf = np.delete(self.skin_surf,np.where(self.skin_surf[:,1]>ymax),axis=0)
            self.skin_surf = np.delete(self.skin_surf,np.where(self.skin_surf[:,0]<xmin),axis=0)

            self.cropped_img = self.cropped_img #[:,xmin:xmax]
            self.x_adj = xmin
            self.y_adj = ymin

            redo = False
            return redo

        def show_plot(self):
            plt.imshow(self.img, cmap='gray')
            if self.skip == False:
                plt.scatter(self.dorsal_pts[:,0],self.dorsal_pts[:,1],color='blue',s=0.5)
            plt.scatter(self.pts_ICS[:,0],self.pts_ICS[:,1], color='green', s=0.3 )
            plt.scatter(self.current_pts_ICS[:,0],self.current_pts_ICS[:,1], color='purple', s=0.3 )
            def press_redo(event):
                self.redo = True 
                print("redo: redo button pressed")
            def press_skip_d(event):
                self.skip_d = True 
                print("skipping. no dorsal points extracted.")
            def press_skip_v(event):
                self.skip_v = True 
                print("skipping. no dorsal points extracted.")
            axes = plt.axes([0.7, 0.05, 0.1, 0.075])
            bredo = matplotlib.widgets.Button(axes,'Redo')
            bredo.on_clicked(press_redo)
            axes = plt.axes([0.4, 0.05, 0.2, 0.075])
            bskip_d = matplotlib.widgets.Button(axes,'Skip dorsal points')
            bskip_d.on_clicked(press_skip_d)
            axes = plt.axes([0.1, 0.05, 0.2, 0.075])
            bskip_v = matplotlib.widgets.Button(axes,'Skip volar points')
            bskip_v.on_clicked(press_skip_v)
            plt.show()
        
        def apply_hyst_thresh(self):
            img = self.img
            #apply hysteresis threshold filter
            hyst = filters.apply_hysteresis_threshold(img, self.hyst_lowt, self.hyst_hight)

            #remove bright noise
            footprint = morphology.disk(1)
            res_bright = morphology.white_tophat(hyst, footprint)
            indices_bright = np.where(res_bright == True)
            indices_bright = np.stack([indices_bright[0],indices_bright[1]],axis=1)
            for i in indices_bright:
                hyst[i[0],i[1]] = False

            #remove dark noise
            hyst_invert = ~hyst
            footprint = morphology.disk(4)
            res_dark = morphology.white_tophat(hyst_invert, footprint)
            indices_dark = np.where(res_dark == True)
            indices_dark = np.stack([indices_dark[0],indices_dark[1]],axis=1)
            for i in indices_dark:
                hyst[i[0],i[1]] = True

            #plt.imshow(hyst)
            #plt.show()

            return hyst



        def ICS_to_RCS(self,points):
            TRI = self.TRI
            pts_ICS = points * 0.066666666667
            for point in pts_ICS:
                point = np.append(point,[0,1])
                point_RCS = np.delete(np.matmul(TRI,point),3)
                try:
                    pts_RCS = np.vstack([pts_RCS,point_RCS])
                except:
                    pts_RCS = np.array(point_RCS,ndmin=2)

            return pts_RCS





def click_point_coordinate(img):
    #setting up a tkinter canvas
    root = Tk()
    canvas = Canvas(root,width = img.shape[1], height = img.shape[0])

    #adding the image
    clicky_img = ImageTk.PhotoImage(image=Image.fromarray(img))
    label = Label(image=clicky_img)
    label.image = clicky_img # keep a reference!
    label.pack()

    canvas.create_image(0,0,image=clicky_img,anchor="nw")

  
    #function to be called when mouse is clicked
    def printcoords(event):
        global points
        #outputting x and y coords to console
        point = np.array([event.x,event.y],ndmin=2)
        try:
            points.append(point)
        except: 
            points=[point]

    #mouseclick event
    label.bind("<Button 1>",printcoords)
    label.pack()

    root.mainloop()
    if len(points)<1:
        skip = True
        return np.array([-1,-1],ndmin=2)

    points_clicked = np.array(points[0],ndmin=2)
    points.pop(0)
    for i in range(0,len(points)):
        points_clicked = np.vstack([points_clicked,points[0]])
        points.pop(0)

    return points_clicked



def binary_to_gs(img):
    #convert edges to GrayScale
    img_gs = np.zeros([img.shape[0],img.shape[1]])
    i=0
    for row in img:
        j=0
        for pixel in row:
            if pixel == True:
                img_gs[i,j] = 255
            else:
                img_gs[i,j] = 0
            j+=1
        i+=1

    return img_gs

def array_to_xy(img,val):
    #convert table data to a two column array of xy values
    i=0
    for row in img:
        j=0
        for pixel in row:
            if pixel == val:
                xy = np.array([j,i],ndmin=2)
                try:
                    xy_coords = np.vstack([xy_coords,xy])
                except:
                    xy_coords = xy
            j+=1
        i+=1
    try:
        xy_coords
    except:
        xy_coords = np.array([0,0],ndmin=2)
    return xy_coords

def get_TRI_array(TRE_folder,TEI_filename):
    TEI = np.loadtxt(TEI_filename, dtype=float)
    TRI_list=[]
    for i in range(0,4):
        TRE_filename = TRE_folder + 'TRE_' + str(i) +'.txt'
        TRE_data = np.loadtxt(TRE_filename, dtype=float)
        n_frames = int(TRE_data.shape[0]/4)
        TRI_scan = np.array(np.matmul(TRE_data[0:4],TEI),ndmin=3)
        
        for i in range(1,n_frames):
            TRE = TRE_data[i*4:i*4+4]
            #TRE = np.matmul(T0n,TRE)
            TRI = np.array(np.matmul(TRE,TEI),ndmin=3)
            TRI_scan = np.concatenate((TRI_scan,TRI),axis=0)
        TRI_list.append(TRI_scan) 
  
    TRI_all = np.stack((TRI_list[0],TRI_list[1],TRI_list[2],TRI_list[3]),axis=0)
    
    return TRI_all


subject_path = 'C:\\Users\\jocel\\OneDrive\\Desktop\\20220607 Formal Data Collection\\Sub00000021\\Trial 1\\'
TEI_path = 'C:\\Users\\jocel\\OneDrive\\Desktop\\20220607 Formal Data Collection\\Sub00000021\\TEI.txt'
APB_path = subject_path + 'APB Dense Point Cloud.txt'
FPB_path = subject_path + 'FPB Dense Point Cloud.txt'
OPP_path = subject_path + 'OPP Dense Point Cloud.txt'

#4d array of all TRIs
TRI_all = get_TRI_array(subject_path,TEI_path)
try:
    APB = np.loadtxt(APB_path)
except:
    APB = np.array([0,0,0],ndmin=2)
try:
    OPP = np.loadtxt(OPP_path)
except:
    OPP = np.array([0,0,0],ndmin=2)
try:
    FPB = np.loadtxt(FPB_path)
except:
    FPB = np.array([0,0,0],ndmin=2)

  
muscles = [APB,FPB,OPP]
s0=Scan('0',12,48,[15,35]) 
#s1=Scan('1',5,57,[10,55]) 
s2=Scan('2',0,61,[0,55])
#s3=Scan('3',19,62,[20,45]) 
scans = [s0,s2]
for scan in [s2,s0]:
    print(scan)
    scan.skin_reconstruct()
try:
    dorsal_pts = np.loadtxt(subject_path + muscle + side + 'points.txt')
    current_muscle = dorsal_pts
except:
    current_muscle = np.array([0,0,0],ndmin = 2)

#current_muscle = np.vstack([volar_pts,dorsal_pts])
#current_muscle = np.loadtxt(subject_path + 'FPBDHdorsalpoints.txt')

for scan in scans:
    print(current_muscle,current_muscle.shape[0])
    scan.shift = (scan.max_skin_point-s0.max_skin_point)
    print(scan.shift)
    scan.segment_images()
    scan.shift_points(s0.max_skin_point)
    np.savetxt(subject_path +str(scan.folder)+ '\\' + muscle + side + 'points.txt',scan.dorsal_points)
    try:
        dorsal_pts = np.vstack([dorsal_pts,scan.dorsal_points])
    except:
        dorsal_pts = scan.dorsal_points
    current_muscle = np.vstack([current_muscle,dorsal_pts])


np.savetxt(subject_path + muscle + side + 'points.txt',dorsal_pts)