# DTN-dynamic
## Train the model

1. The dataset to train real-time model in /N_Prediction/data/historical/rawdata/nomaldata/15s. First you need run BILSTM.py to train the model to get the first real-time prediction model. Then you can use the increase_BILSTM.py to train the model with incremental training.
2. To train the optimization model we need to combine all data. The dataset in /N_Prediction/data/historical/rawdata/packtlossdata/all.csv. You can use N_XGBoost.py to train the model. 

## Do the transfer

1. When you run simple_dtnaas_controllerprp.py, initially set the value of the sequence to the same value. During the transfer, a new folder will be created based on the value of i to store the data collected in real time, as well as the predicted data. During the transfer it will call real-time.py and N_XGBoost.py to make the prediction. 
2. After the transfer, you can find the real-time data and predicted data in the /N_Prediction/data/real-time.

## Process the transfer data

1. testsocre.py is used to calculate RMSE. 
2. getmean.py is used to calculate the average value of each parameter in multiple transfers. 
3. draw.py is used to draw bar charts. 
4. drawbox.py is used to draw a box plot of time.

## Other folders

1. About 8,10,...,90, these are the data corresponding to each transfer_id extracted from Promethues, which can be used to evaluate the results.
2. For the model folder, it stores the models for the first prediction and the second prediction.

