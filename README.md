Topic: Sentiment Analysis on Movie Reviews Team: Anirban, Murtaza and Swarna

We have used the following Python packages in our code, which need to be installed first: pandas, numpy, scikit-learn, Beautiful Soup, NLTK.
There are two methods of installing these modules and getting the code up and running.

1.	The installation process can be done using a software called pip, which can automatically download and 	install Python modules for us. The easiest way to install pip is through the use of a python program called 	get-pip.py, located at https://bootstrap.pypa.io/get-pip.py . Save this page under its default name get-	pip.py.
Next, open windows command prompt and navigate to the directory where you saved the get-pip.py file in the above step and then run the following command.
python get-pip.py to install pip
That completes the pip installation now all the above mentioned required python modules can be easily installed using the necessary pip command which is usually in the following format
pip install beautifulsoup4
We however need to search for the exact pip command for each module in their respective documentations. To avoid the hassle we in fact used the second enlisted method.

2.	Alternately we can install a software ‘Enthought Canopy’ which is a comprehensive Python analysis 	environment with integrated IDE, Compiler, Modules Installer etc. It can be downloaded here - 	https://www.enthought.com/products/canopy/
After installing the software, open the ‘Package Manager’ from the welcome page of the software. Look up the required modules using the search feature and install them with one click.
Now we are ready to run the code. Download training data.csv and test data.csv from https://drive.google.com/file/d/0Bz4sXh4nTtM9YlhYVWtmV0Fhenc/view?usp=sharing
https://drive.google.com/file/d/0Bz4sXh4nTtM9U010emxWTy1ZOXc/view?usp=sharing 
and put them in one folder along with the 'Sentiment Analysis.py' file, and then run the Sentiment Analysis.py file. If you chose the Canopy installation procedure, this will be easy as opening the file using the editor and clicking on the green run button in the toolbar.

Note: 
1)The above links need to be used to download the data file instead of the data provided in the Kaggle competition website because our code uses a single file with all the movie reviews instead of individual txt files for each movie review.
2)After first run of the Sentiment Analysis.py code, line number 11 with the code ‘nltk.download()’ can be commented to prevent the repeated popup of the download window during every execution. The download needs to be performed once. 

Project Report Webpage Link: http://www.cs.uml.edu/~svedam/MachineLearning/ProjectReport.html 

