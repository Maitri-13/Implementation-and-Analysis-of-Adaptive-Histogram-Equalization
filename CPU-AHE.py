#######################################################
#Author		: Subhradeep Dutta & Maitri Chattopadhyay
#Course Name:Advanced Computer Architecure (ECEN 5593)
#Description: This program implements the Adaptive
#			  Histogram Equalization algorithm on the
#			  CPU and then calls another function for
#			  the GPU. It uses Python's time module to
#			  compute the execution time for each case
#Prerequisites: Python 2.7
#				OpenCV 3.x
#				PyCUDA
#######################################################
import time
import numpy as np
import cv2
from GPU_AHE import ahe

#define the window size
kernel_val = raw_input("Enter the size of the kernel")
print "Kernel Size is ", kernel_val
kernel_size = int(kernel_val)
kernel_size_squared = int(kernel_size * kernel_size)

#read the image to be modified
image = cv2.imread('Chest.jpg',0)
#Pad the image on the edges
img = np.lib.pad(image,(kernel_size,kernel_size),'reflect')

#Calculate the dimensions of the image
img_size=img.shape
print 'The size of the image is ', img_size
#Maximum intensity for a 8 bit grayscale image
max_intensity = 255
#Create an empty array like the original image to store the new image
ime_ahe_cpu = np.zeros_like(img)

# Timer starts here
t_start = time.time()
print 'Start Time ',t_start

#####################CPU LOGIC#####################

#Iterate through the pixels in the image
for i in xrange(0,img_size[0]-kernel_size):
    for j in xrange(0,img_size[1]-kernel_size):
		#Extract a window from the image
        kernel = img[i:i+kernel_size,j:j+kernel_size]
		#Sort the extracted window pixels
        kernel_flat = np.sort(kernel.flatten())
        # Calculate the rank of the pixel
        rank = np.where(kernel_flat == img[i,j])[0][0]
		#Write the value of the new pixel to the other array
        ime_ahe_cpu[i,j] = int((rank * max_intensity )/(kernel_size_squared))
###################################################

# Timer ends here
t_end = time.time()
print 'End Time ',t_end
print 'Total time taken in seconds for the CPU ',(t_end-t_start)
#Slice the image exlcuding the padded part
img_sliced_cpu = ime_ahe_cpu[(0+kernel_size):(img_size[0]-kernel_size),(0+kernel_size):(img_size[1]-kernel_size)]
image_output_cpu = np.array(img_sliced_cpu, dtype = np.uint8)

#Write the image back to file
cv2.imwrite('Output_CPU-'+str(kernel_size)+'-'+str(t_end-t_start)+'.png',image_output_cpu)


##############  GPU Section #####################
ime_ahe_gpu = np.zeros_like(img)
t_start = time.time()
#Timer starts here
print 'Start Time ',t_start
ime_ahe_gpu = ahe(img, kernel_size, max_intensity)
t_end = time.time()
#Timer ends here
print 'End Time ',t_end
print 'Total time taken in seconds for the GPU ',(t_end-t_start)
img_sliced_gpu = ime_ahe_gpu[(0+kernel_size):(img_size[0]-kernel_size),(0+kernel_size):(img_size[1]-kernel_size)]
#Slice the image exlcuding the padded part
image_output_gpu = np.array(img_sliced_gpu, dtype = np.uint8)
#Write the image back to file
cv2.imwrite('Output_GPU-'+str(kernel_size)+'-'+str(t_end-t_start)+'.png',image_output_gpu)


#Calculate the difference between the CPU & GPU version
diff = cv2.subtract(image_output_cpu, image_output_cpu)
#Write the difference in the CPU and GPU versions into an image
cv2.imwrite('diff.png', diff)