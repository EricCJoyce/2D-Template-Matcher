#  Eric C Joyce, Stevens Institute of Technology, 2019

#  Find the template object in the given image.
#  Use command-line parameters or default values to cover a range of scales and a range of stretches.

#   argv[0] = template.py
#   argv[1] = search image
#  {argv[2..n] = flags}

#  e.g.
#  python template.py searchme.png -v

import sys															#  Receive command line arguments
import os															#  Search file system
import cv2															#  OpenCV
import numpy as np													#  NumPy
import imutils														#  Used for stretching and scaling

def main():
	params = parseRunParameters()									#  Get command-line options
	if params['helpme']:											#  Did user ask for help?
		usage()														#  Display options
		return

	if len(sys.argv) < 2:  ##########################################  Step 1: check arguments and files
		usage()
		return

	if not os.path.exists(sys.argv[1]):								#  Must have an image file in which we search
		print('Unable to find image file "' + sys.argv[1] + '"')
		return
																	#  Acceptable OpenCV image types
	imgExt = ['pbm', 'pgm', 'ppm', 'jpg', 'jpeg', 'jpe', 'jp2',  'png', 'bmp', 'tiff', 'tif', 'sr', 'ras']
	templateList = []												#  Filepaths for template(s)
	if os.path.exists(params['template']):							#  The path identified was found
		if os.path.isdir(params['template']):						#  It's a directory, and it exists.
			if params['verbose']:
				print('Searching template directory, "' + params['template'] + '":')
			for f in os.listdir(params['template']):				#  Load all image files
				arr = f.split('.')
				if arr[-1].lower() in imgExt:
					if params['verbose']:
						print('  ' + f)
					templateList.append(params['template'] + '/' + f)
		elif os.path.isfile(params['template']):					#  It's a single file, and it exists.
			if params['verbose']:
				print('Found template file, "' + params['template'] + '"')
			templateList.append(params['template'])
	elif params['template'] == 'template.*':						#  Either default parameter 'template.*' was used,
		for ext in imgExt:											#  and we must try extensions...
			f = params['template'][:]
			f.replace('*', ext)
			if os.path.exists(f):
				if params['verbose']:
					print('Found "' + f + '"')
				templateList.append(f)
	else:															#  ...or the given argument claimed was not found.
		print('Unable to find "' + params['template'] + '"')
		return

	imgRGB = cv2.imread(sys.argv[1])	#############################  Step 2: load the image in which we search
	imgGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)				#  Convert to grayscale
	imgW, imgH = imgGray.shape[::-1]								#  Save world-image dimensions
	imgGray = cv2.GaussianBlur(imgGray, (5, 5), 3)					#  Blur slightly

	if params['verbose']:
		print('Settings')
		print('         Image:  ' + str(imgW) + ' x ' + str(imgH))
		if params['nonmaxsupp'] is not None:
			print('  Non-Max.Supp:  ' + str(params['nonmaxsupp']))
		else:
			print('  Non-Max.Supp:  None')
		print('         Scale:  ' + str(params['s0']) + ' --> ' + str(params['s1']))
		print('         steps:  ' + str(params['sn']))
		print('  Aspect Ratio: ' + str(-params['flex']) + ' --> ' + str(params['flex']))
		print('         steps:  ' + str(params['flexsteps']))
		if params['threshold'] is not None:
			print('     Threshold:  ' + str(params['threshold']))
		else:
			print('  No threshold')
		print('\nStarting search\n')

	acceptable = []													#  Save matches meeting our criteria
	found = None

	for templateFN in templateList:	#################################  Step 3: search for each template file listed

		template = cv2.imread(templateFN)							#  Load the template file
		templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)	#  Convert to grayscale
		templateW, templateH = templateGray.shape[::-1]				#  Save template dimensions
		templateGray = cv2.GaussianBlur(templateGray, (5, 5), 3)	#  Blur slightly
																	#  Try several scales over several steps
		for scale in np.linspace(params['s0'], params['s1'], params['sn'])[::-1]:
																	#  Try different aspect ratios
			for distort in np.linspace(-params['flex'], params['flex'], params['flexsteps']):

				resized = imutils.resize(templateGray, width=int(templateW * scale * (1.0 + distort)), \
				                                       height=int(templateH * scale))
				resizedW, resizedH = resized.shape[::-1]

				if params['verbose']:
					outputstring = 'Scale: ' + str(scale) + ', Aspect ratio: ' + str(distort) +', resized template ' + str(resizedW) + ' x ' + str(resizedH)

				if resizedW <= imgW and resizedH <= imgH:
					result = cv2.matchTemplate(imgGray, resized, cv2.TM_CCOEFF_NORMED)

					if params['threshold'] is not None:	#############  Threshold given: find matches above or equal
																	#  where() returns an array of y's and an array of x's
						loc = np.where(result >= params['threshold'])
						loc = zip(loc[1], loc[0])					#  Convert from numpy (y, x) to opencv (x, y)
						if params['verbose']:
							print(outputstring)
						for i in range(0, len(loc)):
							upperleft = (int(loc[i][0]), int(loc[i][1]))
							lowerright = (int(loc[i][0] + templateW * scale * (1.0 + distort)), int(loc[i][1] + templateH * scale))
							found = (result[loc[i][1]][loc[i][0]], upperleft, lowerright)
							acceptable.append( found )
							if params['verbose']:
								sys.stdout.write('  ' + str(i) + '\r')
								sys.stdout.flush()
						if params['verbose'] and len(loc) > 0:
							print('')

					else:	#########################################  No threshold: just find the best
						(_, maxVal, _, upperleft) = cv2.minMaxLoc(result)

						if found is None or maxVal > acceptable[0][0]:
							lowerright = (int(upperleft[0] + templateW * scale * (1.0 + distort)), int(upperleft[1] + templateH * scale))
							found = (maxVal, upperleft, lowerright)
							acceptable = [ found ]
							if params['verbose']:
								outputstring += '\t*'

						if params['verbose']:
							print(outputstring)

			if params['verbose'] and params['threshold'] is not None:
				print('')

	fh = open('template.log', 'w')	#################################  Step 4: output finds to log file (or render)
	fh.write(sys.argv[1] + '\n')									#  First line is the search-image file, for reference
	if params['nonmaxsupp'] is not None:
		if params['verbose']:
			print('Applying non-maximum suppression:')
		acceptable = nonmaxsupp(acceptable, params)					#  Apply non-max. suppression
	for i in range(0, len(acceptable)):								#  For every match found
		(confidence, upperleft, lowerright) = acceptable[i]			#  unpack corners and confidence
		fstr  = str(upperleft[0]) + ' ' + str(upperleft[1]) + ' '	#  Build a string to write to the log file
		fstr += str(lowerright[0]) + ' ' + str(lowerright[1]) + ' '
		fstr += str(confidence) + '\n'
		fh.write(fstr)
		if params['render']:										#  If rendering, draw a rectangle
			if params['nonmaxsupp'] is None:						#  No non-max? Draw red
				cv2.rectangle(imgRGB, upperleft, lowerright, (0, 0, 255), 1)
			else:													#  Using non-max? Draw green
				cv2.rectangle(imgRGB, upperleft, lowerright, (0, 255, 0), 1)
	fh.close()

	if params['render']:											#  If we used a threshold, name the render file
		arr = sys.argv[1].split('.')								#  using that threshold for reference
		if params['threshold'] is not None:
			cv2.imwrite(arr[0] + '_optimum_' + str(params['threshold']) + '.png', imgRGB)
		else:
			cv2.imwrite(arr[0] + '_optimum.png', imgRGB)

	return

#  If we have been asked to use non-maximum suppression, then filter the candidate
#  matches here and return a new list. Intuitively, we replace rectangles overlapping
#  by more than 'overlap' with .
#  Each in 'finds' is of the form (score, upperleft, lowerright).
#  Score is a float, upperleft and lowerright are both tuples, (int, int)
def nonmaxsupp(finds, params):
	survivors = []													#  Prepare list of winnowed-down matches
	upperleftX = []													#  Prepare list of upper-left X components
	upperleftY = []													#  Prepare list of upper-left Y components
	lowerrightX = []												#  Prepare list of lower-right X components
	lowerrightY = []												#  Prepare list of lower-right X components
	areas = []														#  Prepare list of bounding-box areas
	for i in range(0, len(finds)):									#  Convert to floats
		upperleftX.append(float(finds[i][1][0]))
		upperleftY.append(float(finds[i][1][1]))
		lowerrightX.append(float(finds[i][2][0]))
		lowerrightY.append(float(finds[i][2][1]))
		areas.append((lowerrightX[-1] - upperleftX[-1] + 1) * (lowerrightY[-1] - upperleftY[-1] + 1))
	upperleftX  = np.array(upperleftX)								#  Convert to numpy arrays
	upperleftY  = np.array(upperleftY)
	lowerrightX = np.array(lowerrightX)
	lowerrightY = np.array(lowerrightY)
	areas       = np.array(areas)
	indices     = np.argsort(lowerrightY)							#  Array of indices sorted according to their
																	#  corresponding bounding-box's lower-right lowness
	while len(indices) > 0:											#  While anything remains...
		if params['verbose']:
			sys.stdout.write('  ' + str( int(round((1.0 - float(len(indices)) / float(len(finds))) * 100))) + '%     \r')
			sys.stdout.flush()
		last = len(indices) - 1										#  Take the last (largest) index
		i = indices[last]
		survivors.append(i)
																	#  Compute overlap
		xx1 = np.maximum(upperleftX[i],  upperleftX[indices[:last]])
		yy1 = np.maximum(upperleftY[i],  upperleftY[indices[:last]])
		xx2 = np.minimum(lowerrightX[i], lowerrightX[indices[:last]])
		yy2 = np.minimum(lowerrightY[i], lowerrightY[indices[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / areas[indices[:last]]

		indices = np.delete(indices, np.concatenate(([last], np.where(overlap > params['nonmaxsupp'])[0])))

	if params['verbose']:
		print('')

	return [finds[x] for x in survivors]

#  Parse the command line and set variables accordingly
def parseRunParameters():
	templatePath = None												#  Where to find info for the template we want to find
	nonmaxsupp = None												#  Whether to use non-maximum suppression
	threshold = None												#  Whether we use a match confidence threshold
	flex = None														#  Imbalance margin to original aspect ratio
	flexsteps = None												#  Degress to this imbalance
	startScale = None												#  Smallest (uniform) scale to try
	endScale = None													#  Largest (uniform) scale to try
	scaleSteps = None												#  Divisions of the scales tried
	verbose = False													#  How talkative should the script be
	render = False													#  Render an image with detection rectangles
	helpme = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-t', '-m', '-th', '-f', '-n', '-s0', '-s1', '-sn', '-v', '-r', '-?', '-help', '--help']
	for i in range(2, len(sys.argv)):
		if sys.argv[i] in flags:
			argtarget = sys.argv[i]
			if argtarget == '-v':
				verbose = True
			elif argtarget == '-r':
				render = True
			elif argtarget == '-?' or argtarget == '-help' or argtarget == '--help':
				helpme = True
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-t':								#  Following argument sets template filepath
					templatePath = argval
				elif argtarget == '-m':								#  Following argument sets suppression threshold
					nonmaxsupp = float(argval)
				elif argtarget == '-th':							#  Following argument sets threshold
					threshold = float(argval)
				elif argtarget == '-f':								#  Following argument sets flex margin
					flex = abs(float(argval))
				elif argtarget == '-n':								#  Following argument sets flex steps
					flexsteps = abs(int(argval))
				elif argtarget == '-s0':							#  Following argument sets starting scale
					startScale = float(argval)
				elif argtarget == '-s1':							#  Following argument sets final scale
					endScale = float(argval)
				elif argtarget == '-sn':							#  Following argument sets scaling steps
					scaleSteps = int(argval)
																	#  Use default values where necessary.
	if templatePath is None:										#  Default to an image file named "template.*" in cwd
		templatePath = 'template.*'
	if flex is None:												#  Default to 0.0 (no flex search)
		flex = 0.0
	if flexsteps is None:											#  Default to one pass if we're not stretching at all
		if flex == 0.0:
			flexsteps = 1
		else:														#  Default to three passes if steps were unspecified:
			flexsteps = 3											#  low end, middle, high-end

	if startScale is None:											#  Default to 0.8
		startScale = 0.8
	if endScale is None:											#  Default to 5.0
		endScale = 5.0
	if scaleSteps is None:											#  Default to 50 steps
		scaleSteps = 50

	params = {}
	params['template'] = templatePath
	params['nonmaxsupp'] = nonmaxsupp
	params['threshold'] = threshold
	params['flex'] = flex
	params['flexsteps'] = flexsteps
	params['s0'] = startScale
	params['s1'] = endScale
	params['sn'] = scaleSteps
	params['verbose'] = verbose
	params['render'] = render
	params['helpme'] = helpme

	return params

#  Let the user know what options this script offers.
def usage():
	print('Usage:  python template.py search-image-filename <options, each preceded by a flag>')
	print(' e.g.:  python template.py basement.jpg -t 0.5 -v')
	print('Flags:  -t  following argument is the filesystem path to the template:')
	print('            if this is an image file, we search using only that image file;')
	print('            if this is a directory, we search using all images in that directory.')
	print('            (By default, script assumes a single image file named "template" in')
	print('            the script\'s directory.)')
	print('        -th following argument is the threshold for detection')
	print('            (omitting a threshold will find the optimal match)')
	print('        -m  following argument is the non-maximum suppression factor')
	print('            (only applicable if you use a threshold)')
	print('        -f  following argument is the aspect ratio "flex" to either side')
	print('        -n  following argument is the number of aspect ratio steps')
	print('        -s0 following argument is the starting uniform scale')
	print('        -s1 following argument is the ending uniform scale')
	print('        -sn following argument is the number of uniform scale steps')
	print('        -v  enable verbosity')
	return

if __name__ == '__main__':
	main()
