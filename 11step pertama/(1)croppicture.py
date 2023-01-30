# demonstrate face detection on 5 Celebrity Faces Dataset
from email.mime import image
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2

mydir = 'validfoto/charlie'  #tempat simpan foto hasil crop
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	im_rgb = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
	cv2.imwrite('{0}/doono{1}.jpg'.format(mydir,i), im_rgb) #ubah namafile data hasil crop
	return face_array
 
# specify folder to plot
folder = 'fotodisimpandulu/carlirusak/' # foto yang akan di crop
i = 1
# enumerate files
for filename in listdir(folder):
	# path
	path = folder + filename
	# get face
	face= extract_face(path)
	print(i, face.shape)
	# plot
	pyplot.subplot(2, 5, i) #jumlah yang bisa di crop berdasarkan ini, (2,5),ini yg ditampilkan nanti2 baris 5 kolom, begitupun seterusnya.
	pyplot.axis('off')
	pyplot.imshow(face)
	# cv2.imshow('s',face)
	# pyplot.savefig()
	# face.save(r'{0}.jpg'.format(i))
	i += 1
	# pyplot.savefig(r'figure{0}.jpg'.format(i))	
pyplot.show()
	
	
