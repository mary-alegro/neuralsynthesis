import glob
import cv2

def main():

    new_size = (480,256) #height,width

    files = glob.glob('/Users/maryana/Projects/neural3d/results/test/*.png')
    files.sort()
    img_array = []
    for f in files:
        img = cv2.imread(f)
        new_img = cv2.resize(img,new_size)
        img_array.append(new_img)

    out = cv2.VideoWriter('/Users/maryana/Projects/neural3d/results/test/bedroom.avi', cv2.VideoWriter_fourcc(*'MJPG'), 5, new_size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



if __name__ == '__main__':
    main()
