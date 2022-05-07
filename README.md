# pervasive_seminar

Install Pupil Desktop Software - https://pupil-labs.com/products/core/
  Make sure WinRAR is installed
Clone the Github Repo - https://github.com/cwittwer/pervasive_seminar.git
	git clone https://github.com/cwittwer/pervasive_seminar.git
Move to that repo
Add the TensorFlow Model Garden - https://github.com/tensorflow/models.git
	git submodule add https://github.com/tensorflow/models.git 
create conda virtual environment(need Anaconda/miniconda)
	conda create -n pervasive_seminar
activate - conda activate pervasive_seminar
Install Dependencies 
	pip install tensorflow
		(if you have gpu use: tensorflow-gpu)
	pip install tensorflow-object-detection-api
		fix the gfile error - https://stackoverflow.com/questions/55591437/attributeerror-module-tensorflow-has-no-attribute-gfile
	pip install zmq opencv-python 
		may need this(pillow Cython lxml matplotlib contextlib2 tf_slim IPython)
 
 Run python main.py
