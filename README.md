# 2D Template Matching
2D Template Matching code repository. This script attempts to find instances of objects in images. Object instances are represented by one or more template images.

# Dependencies and Set-Up

## NumPy
http://www.numpy.org/ A major organ of OpenCV, used for vector and matrix operations
## OpenCV
https://opencv.org/ The core of all computer vision applications. For this application, the straightforward Python installation will suffice: https://docs.opencv.org/4.1.0/d2/de6/tutorial_py_setup_in_ubuntu.html. Other applications will require compiling C++ using the OpenCV library, and that is a more involved installation. The following worked for us, installing OpenCV 3.1.0 on Ubuntu 16.04:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install git libgtk-3-dev
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libdc1394-22-dev libeigen3-dev libtheora-dev libvorbis-dev
sudo apt-get install libtbb2 libtbb-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev sphinx-common yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libavresample-dev
sudo apt-get install libatlas-base-dev gfortran
```
The foregone commands install all the requisite libraries. Now we install and build OpenCV:
```
sudo -s
cd /opt
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
unzip opencv_contrib.zip
mv /opt/opencv-3.1.0/ /opt/opencv/
mv /opt/opencv_contrib-3.1.0/ /opt/opencv_contrib/
cd opencv
mkdir release
cd release
cmake -D WITH_IPP=ON -D INSTALL_CREATE_DISTRIB=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules /opt/opencv/
make
make install
ldconfig
exit
cd ~
```
If all went well, we should be able to query the version of the OpenCV installation:
```
pkg-config --modversion opencv
```
We should also be able to compile C++ code that uses OpenCV utilities. (Suppose we've written some such program, `helloworld.cpp`.)
```
g++ helloworld.cpp -o helloworld `pkg-config --cflags --libs opencv`
```
## imutils
https://github.com/jrosebr1/imutils Image manipulation tools, used here for resizing and scaling

# Inputs

## Example Script Call
The only required argument is an image file in which to search, as seen here: `python template.py haystack.png`
If a "needle" is not specified, the script will attempt to find a compatible image file in its own directory named "template". Acceptable file types are:
* .pbm
* .pgm
* .ppm
* .jpg
* .jpeg
* .jpe
* .jp2
* .png
* .bmp
* .tiff
* .tif
* .sr
* .ras

Specify a "needle" using the `-t` flag, followed by the filepath for your needle. The needle can be a single image or a directory. If the needle is an image, then performance is straightforward: the script attempts to find instances of this single needle in your haystack. If the needle is a directory, then that directory will be assumed to contain at least one image that can act as a template. The idea here is to include images of your needle from different angles, different perspectives, to increase the likelihood of its being found in your haystack. However, this approach is also likely to increase your false positives, too.

Other adjustments to script performance are specified using other command line flags. Please see these described below in "Parameters."

# Outputs

## The Log File
The file, `template.log`, is a text record of all matches generated by the most recent call to `template.py`. Its first line is the name of the haystack file. Every following line has the following data:
* X coordinate of the match's upper-left corner (integer)
* Y coordinate of the match's upper-left corner (integer)
* X coordinate of the match's lower-right corner (integer)
* Y coordinate of the match's lower-right corner (integer)
* The confidence score of this match (real)
There is one line for every match found. By default, the script will only find one, optimal match. Including other matches depends on the particular allowances you determine using the command-line arguments.

## Optional Rendering
You can optionally render matches to new image files. Please see below, the section detailing the `-r` parameter.

# Parameters

## `-t` Set the Template
As described above, the script allows omission of a template, your needle, to find in the given haystack image. If you want to specify a template for the search, use this flag to let the script know that the following argument is the file or directory it should use.
`python template.py haystack.png -t needle.jpg`
When the arguemtn following `-t` is a directory, the script will use all image files in this directory as templates.
`python template.py haystack.png -t objectsToFind/inMyBasement/needle`
In this case, the directory "needle" might contain several views of the same object, allowing that it might appear variously in the "haystack" image.
## `-m` Set Non-Maximum Suppression
Often an object will excite several, clustered responses from the matcher. This means that a match has been made, then again one pixel over, and again one pixel over from that, etc. Even when the matches are correct, this can be more redundant information that you need. Non-maximum suppression shuts down (suppresses) all but the best match within overlapping clusters. The flag for this argument should be followed by a real number in [0.0, 1.0] inclusive indicating how much overlap to consider permissible. Closer to 0.0 will drop more redundant positives; closer to 1.0 will include more redundant positives.
## `-th` Set the Threshold
When this argument is unspecified, the script finds a single, most confident match. Specifying a threshold means it will record all candidate matches with confidence scores greater or equal to this threshold. The parameter must be a real number in [0.0, 1.0] inclusive (though these extrema are unlikely to be helpful.) Numbers closer to 0.0 will admit more possibilities (and several false positives); numbers closer to 1.0 demand more confident matches and may omit actual matches. Use this parameter in conjunction with non-maximum suppression to winnow down clustered matches to single matches.
## `-f` Set Aspect Ratio Flexibility
The script iterates over a range of scales searching for needle. The `-f` flag and its accompanying `-t` allow further that needle may appear in haystack vertically or horizontally distorted. This seemed a sensible addition when we were experimenting with the fusebox. We had one template image of a particular fusebox, but wanted to use the same template to find other fuseboxes that were similar but taller. The argument following `-f` should be a real number in [0.0, 1.0] inclusive. This number is the margin for both vertical and horizontal distortion to be covered. If you want to find needles that may be twice as tall or twice as wide as your template needle, pass `-f 1.0`.
## `-n` Set Aspect Ratio Gradation
This argument specifies how many iterations should separate the vertical extreme from the horizontal extreme specified by `-f`. By default, `n` becomes 3 to cover the vertical extreme, the undistorted middle, and the horizontal extreme. `n` only takes effect if an `f` has been specified.
## `-s0` Set Starting and the `-s1` Ending Scale
The script iterates over a range of scales, searching for needle at, say, half the size of the template image, up to twice the size of the template image. You can specify this range using the `-s0` and `-s1` parameters. The search just mentioned could be called thus:
`python template.py haystack.png -s0 0.5 -s1 2.0`
## `-sn` Set Scale Gradation
This argument sets the number of steps to take from the starting scale to the ending scale. The finer the grain of this search, the more subtleties you can capture, and the more time your search will take.
## `-v` Enable Verbosity
Template matching can be time-consuming. It's often helpful for the program to show signs of life. No argument follows `-v`.
## `-r` Render Finds to an Image
Your purposes may not require rendering matches to image--all the important information is written to the log file. However, it is sometimes helpful to have some visual indicators of the script's performance. No argument should follow the flag `-r`. A file will be generated and named after your haystack image file and settings.
`python template.py haystack.png -r` will generate a file named `haystack_optimum.png`. If you specify a threshold, then that will be included, too. `python template.py haystack.png -r -th 0.6` will produce `haystack_optimum_0.6.png`.

## `-?`, `-help`, `--help` Help!
Display some notes on how to use this script.

# Recommended Settings

Settings really depend on what you hope to detect. Are your particular needle and haystack conducive to many false positives? Then omit the threshold argument and let the script make a single best guess. Are there likely to be several instances of your needle that you'd like to find? Then allow a threshold generous enough (closer to 0.0) to admit more hypotheses, but not so generous (closer to 1.0) that only verbatim matches will make the cut.

The fusebox needle in our basement experiments proved a difficult case. Typically, thresholds around 0.6 would find something satisfactorily box-like; lower thresholds tended to admit too many hypotheses or to "hallucinate" shapes that seem to match the fusebox template's distributions of light and dark.