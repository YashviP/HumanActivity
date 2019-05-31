import numpy as np
import cv2
import os
import math
import time
import pickle

#handclapping/person01_handclapping_d1_uncomp.avi


def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default


def lbp(frame_lbp):
	#img=cv2.imread(frame_lbp)
	print(frame_lbp.shape)
	img=cv2.cvtColor(frame_lbp, cv2.COLOR_BGR2GRAY)
	transformed_img =cv2.cvtColor(frame_lbp, cv2.COLOR_BGR2GRAY)
	for x in range(0, len(img)):
		for y in range(0, len(img[0])):
			center = img[x,y]
			top_left = get_pixel_else_0(img, x - 1, y - 1)
			top_up = get_pixel_else_0(img, x, y - 1)
			top_right = get_pixel_else_0(img, x + 1, y - 1)
			right = get_pixel_else_0(img, x + 1, y)
			left = get_pixel_else_0(img, x - 1, y)
			bottom_left = get_pixel_else_0(img, x - 1, y + 1)
			bottom_right = get_pixel_else_0(img, x + 1, y + 1)
			bottom_down = get_pixel_else_0(img, x, y + 1)
			values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,bottom_down, bottom_left, left])
			weights = [1, 2, 4, 8, 16, 32, 64, 128]

			res = 0

			for a in range(0, len(values)):
				res += weights[a] * values[a]

			transformed_img.itemset((x, y), res)

		return transformed_img



featuresize=0
def bg_sub(filename,file_no,final_res):
	try:
		#filename='jogging/person16_jogging_d1_uncomp.avi'
		ct=0
		res=[]
		cap = cv2.VideoCapture(filename)
		n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		#print(n_frames)


		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps=int(cap.get(cv2.CAP_PROP_FPS))


		lbp_res=np.ndarray(shape=(5,h,w))

		feature_params = dict(maxCorners=100,
	                      qualityLevel=0.3,
	                      minDistance=7,
	                      blockSize=7)

		# Parameters for lucas kanade optical flow
		lk_params = dict(winSize=(w,h),
	                 maxLevel=2,
	                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		color = np.random.randint(0, 255, (100, 3))



		#fourcc = cv2.VideoWriter_fourcc('*MJPG')
		#videoout = cv2.VideoWriter(out_loc, fourcc, fps, (w, h))
		_, frame= cap.read()
		old_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		start_time=time.time()


		p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)
		mask = np.zeros_like(frame)

		ret, thresh = cv2.threshold(frame, 127, 255, 0)
		fgbg = cv2.createBackgroundSubtractorMOG2()
		for i in range(n_frames - 2):
			ret, frame = cap.read()
			if(ct<5):
				lbp1=lbp(frame)
				#print(lbp1.shape)
				lbp_res[ct]=lbp1

			ct=ct+1
			new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			if ret is False:
				print("Cannot read video stream")
				break
			fgmask = fgbg.apply(new_frame)
			fg_frame = np.float32(fgmask)/255.0

			gx = cv2.Sobel(fg_frame, cv2.CV_32F, 1, 0, ksize=1)
			gy = cv2.Sobel(fg_frame, cv2.CV_32F, 0, 1, ksize=1)


			mag,angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

			if(i==0):
				sum_mag=np.zeros(shape=mag.shape,dtype=mag.dtype)
				sum_angle=np.zeros(shape=angle.shape,dtype=angle.dtype)

				sum_mag=np.add(sum_mag,mag)
				sum_angle=np.add(sum_angle,angle)

			else:
				sum_mag = np.add(sum_mag, mag)
				sum_angle = np.add(sum_angle, angle)

			'''M = cv2.moments(fgmask)
			if i==0:
				init_cX=int(M["m10"] / M["m00"])
				init_cY=int(M["m01"] / M["m00"])
				cv2.circle(fgmask, (init_cX, init_cY), 5, (255, 255, 255), -1)
			else:
				if(M["m00"]==0)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				cv2.circle(fgmask, (cX, cY), 5, (255, 255, 255), -1)
            '''
			p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame,new_frame, p0, None, **lk_params)
			good_new = p1[st == 1]
			good_old = p0[st == 1]

			for j, (new, old) in enumerate(zip(good_new, good_old)):
				a, b = new.ravel()
				c, d = old.ravel()
				cv2.line(fgmask, (a, b), (c, d), color[j].tolist(), 2)
				cv2.circle(fgmask, (a, b), 2, color[j].tolist(), -1)

			#print(angle.shape)
			cv2.imshow('frame', fgmask)
	#	    videoout.write(fgmask)
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		end_time=time.time()
		tot_time=end_time-start_time
		hog_avg_mag=sum_mag/n_frames


		hog_mag=hog_avg_mag.flatten()
		#print(hog_mag.shape)


		#print(hog_avg_mag)
		hog_avg_angle=sum_angle/n_frames

		hog_angle = hog_avg_angle.flatten()
		#print(hog_angle.shape)

		res = hog_mag+hog_angle

		old_sum_X=0
		old_sum_Y=0
		new_sum_X=0
		new_sum_Y=0
		#print(good_new.shape)
		#print(good_old.shape)
		len1=len(good_new)
		for j, (new, old) in enumerate(zip(good_new, good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			old_sum_X = old_sum_X+c
			old_sum_Y = old_sum_Y+d
			new_sum_X = new_sum_X+a
			new_sum_Y = new_sum_Y+b


		old_sum_X=old_sum_X/len1
		old_sum_Y=old_sum_Y/len1
		new_sum_X=new_sum_X/len1
		new_sum_Y=new_sum_Y/len1

		disp_opt=math.sqrt(((old_sum_X-new_sum_X)**2+(old_sum_Y-new_sum_Y)**2))

		#disp=math.sqrt(((init_cX-cX)**2+(init_cY-cY)**2))

		#velocity=disp/tot_time
		velocity_opt=disp_opt/tot_time

		#print(disp)
		#print(disp_opt)
		#print(velocity)
		#print(velocity_opt)

		res = list(res)
		res.append(disp_opt)
		res.append(velocity_opt)

		# lbp_res=np.array(lbp1)

		lbp_res = lbp_res.flatten()
		lbp_res = list(lbp_res)
		# print(lbp_res)
		# print(lbp_res.shape)
		res = [*res, *lbp_res]
		# res=list(res)
		res = np.array(res)
		featuresize = len(res)
		print(featuresize)

		final_res[file_no] = res
		cap.release()
		# videoout.release()
		cv2.destroyAllWindows()
	except LookupError:
		print("Error")



activities=['walking','running','jogging','boxing','handclapping','handwaving']
out_folder='out_bg_sub'



num_files=0
total_files=0


for activity1 in activities:
	files1=os.listdir(activity1)
	total_files=total_files+len(files1)

final_res=np.ndarray(shape=(total_files,115202))
label=np.ndarray(shape=(total_files,1))

for i in range(0,len(activities)):
	print(activities[i])
	files=os.listdir(activities[i])
	for file in files:
		label[num_files]=i;
		input_f=activities[i]+'/'+file
		print(input_f)
		#out_f=out_folder+'/'+activity+'/'+file
		#print(out_f)
		bg_sub(input_f,num_files,final_res)
		print(final_res)
		num_files=num_files+1

print(final_res)
print(final_res.shape)
print(label)
print(label.shape)

with open('feature_mat.pkl','wb') as f:
      pickle.dump(final_res, f)

with open('label.pkl','wb') as f:
      pickle.dump(label, f)
