Working with GitHub Actions and PyPI for butext
===============================================

Like many of Python’s popular libraries, butext is hosted on PyPI, meaning it is pip installable. Whenever a new version of butext is uploaded to PyPI, that version will be the version installed when someone runs the pip installation. The butext library is uploaded automatically to PyPI and TestPyPI whenever a version of the butext project with an increased version number is pushed to github. When bumping up, a maintainer should also make sure to update the install_requires parameter by adding any new libraries that have been implemented into the functions. 

.. code-block:: none

	setup(
	(...)
		version=’0.3.5’
		install_requires=['pandas','numpy','scikit-learn']
		(...)
		)

The setup.py file also contains other parameters which, while not always mandatory, inform the page for the project on https://pypi.org/project/butext/

The uploading is managed through the release.yaml file in the workflows folder, which has three jobs that it performs. The first job is the building of the new distribution, which means the building of the .whl and .tar.gz files (the files that are actually processed when the library is run). 

In the event that a maintainer is unable to update the library from push, it can be done manually. First, run this command in the terminal:
python setup.py sdist bdist_wheel
This creates the aforementioned .whl and .tar.gz files and stores them in the dist (distributions) folder. When ready to upload this to PyPI, run the following command:
twine upload dist/*
You may have to run it as python -m twine upload dist/* depending on your environment. At this point it is possible that you will be asked for authorization, at which point you’ll want to go from the butext home page on PyPI to ‘manage project’ >> ‘settings’ >> ‘Create a token for butext’. This token only appears once, so if you aren’t going to enter it immediately then store it. Some coding environments will not display the token on screen after pasting it. So, if you paste it into the environment and see nothing pasted, try entering it before attempting to paste again.







