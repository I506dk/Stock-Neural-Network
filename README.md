Neural Network to Predict Stock Market Prices
Preface: I am by no means the greatest programmer to ever live. 
My code is rather simplistic and (hopefully) straightforward. 
Any comments/concerns/corrections are welcomed.
Sincerely,
I5


Summary:
This is a Long Short Term Memory Neural Network designed to predict 
future stock market prices. It was created to be rather self using APIs 
keys from Alpha Vantage to download the latest stock data. But it does 
rely on having a history file being already made for each stock symbol 
being used. The history csv file contains all past data, and the network
appends the data it downloads to this file. I downloaded the past five
years worth of data from Yahoo(?) and renamed that file to 
stocksymbol_History.csv for the network to use. Beyond that, the 
network needs nothing more.


Requirements (At the time this was created):
* python 3.8.2
* plotly 4.8.0
* numpy 1.18.4
* pandas 1.0.3
* tensorflow 2.2.0
* keras 2.3.1
* sklearn 0.0
* alpha-vantage 2.2.0
* urllib3 1.24.3


Architecture: 
The model consists of 5 layers. They are as follows LSTM, Dropout, Dense, 
Dropout, and Dense. The LSTM layer has 64 nodes, which I settled on due to 
trial and error. The first dropout layer has a 0.5% dropout. The second has
a 0.1% dropout. I decided that after testing, the first dropout layer needed
to be rather harsh to try and keep the model form overfitting. The second 
layer is really just to back up the first dropout layer and insure 
(or try to) no overfitting. The dense layers are nothing special, the first 
having 64 nodes since it is a hidden layer. The second dense layer only has 
one node since it is the output layer. Sigmoid and Linear activations were 
used as well. I used the adam optimizer with a fairly small learning rate, 
however when using adam it will adjust the learning rate once it starts to 
plateau. I also used a mean squared error to monitor its learning. Early 
stopping was also implemented as a last resort to stop overfitting if any 
had occurred.