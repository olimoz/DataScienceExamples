# Data Science Examples

These examples show how Deep Learning can be applied to investigate financial and performance management issues in SME's.
The list is not exhaustive, especially not for Caret in R (ie Statistical Learning as opposed to Machine Learning)
However, it is a useful oversight of common tools and a code resource for myself

## Examples of work

| Project Name & Code Link                                                                                                                          | Language   | ML Package     | Techniques                                                             |
|---------------------------------------------------------------------------------------------------------------------------------------------------|------------|----------------|------------------------------------------------------------------------|
| [Parts Forecasting SimpleModels](https://github.com/olimoz/DataScienceExamples/blob/master/Parts_Forecasting_SimpleModels.pdf)                    | R          | Caret          | GLM, Treebag, XGBoost                                                  |
| [Parts Forecasting Deep](https://github.com/olimoz/DataScienceExamples/blob/master/Parts_Forecasting_Deep.pdf)                    	            | R & Python | Keras in Python| Wavenet (causal dilated 1dconv)                                        |
| [Parts Forecasting Exploration](https://github.com/olimoz/DataScienceExamples/blob/master/Parts_Forecasting_Exploration.pdf)                      | R          | Keras in R     | Timeseries, ARIMA, TScount, Clustering (wavelet, dtw etc), Word2Vec    |
| [Cash Flow Modelling](https://github.com/olimoz/DataScienceExamples/blob/master/Cash_Flow_Deep.pdf)                            		    | R & Python | Keras in Python| LSTM, Wavenet (causal dilated 1dconv), SQL                             |
| [Accounting Anomaly Detection](https://github.com/olimoz/DataScienceExamples/blob/master/AcctgAnomaly_forPublication.Rmd)                         | R & Python | Keras in Python| Anomaly Detection for FCA compliance. Works really well!               |
| [Washout Chain Modelling](https://github.com/olimoz/DataScienceExamples/blob/master/Washout_Chain_Modelling.pdf)              	            | Python     | Keras          | XBRL data download, RegEx, Wrangling! Variational Autoencoder in Keras |
| [Transformer for Common Reasoning](https://github.com/olimoz/DataScienceExamples/blob/master/TransformerForCommonReasoning.pdf)                   | Python     | TF2 + Pytorch  | Transformer in TF2, BERT in PyTorch, Huggingface Data                  |
| [OOP and AutoEncoding](https://github.com/olimoz/DataScienceExamples/blob/master/CompletingTheCircle_AutoEncoder.pdf)             	            | Python     | TF2            | Convolutional Encoder/Decoder in TF2, AI vs OOP. Work in Progress.     |
| [UK Company Finances, Data Extract and VAE](https://github.com/olimoz/DataScienceExamples/blob/master/CompaniesHs_Step1_DataExtract_and_VAE.py)   | Python     | Keras          | XBRL, RegEx, Parallel Processing, Variational Autoencoder in Keras     |
| [UK Company Finances, AutoEncoder](https://github.com/olimoz/DataScienceExamples/blob/master/CompaniesHs_Step2_AutoEncoder.py)         	    | Python     | Keras          | Autoencoder in Keras, use of latent space                              |
| [UK Company Finances, AutoEncoder Critic](https://github.com/olimoz/DataScienceExamples/blob/master/CompaniesHs_Step2_AutoEncoder_2Inputs.py)     | Python     | Keras          | Autoencoder, with my own novel architecture                            |
| [UK Company Finances, AutoEncoder 2Inputs](https://github.com/olimoz/DataScienceExamples/blob/master/CompaniesHs_Step2_AutoEncoder_Critic.py)     | Python     | Keras          | Using Two AutoEncoders for Anomaly Detection (Reconstruction Errors)   |
| [UK Company Finances, VAE_MMD](https://github.com/olimoz/DataScienceExamples/blob/master/CompaniesHs_Step3_VAE_MMD.py)              	            | Python     | TF2            | MMD VAE in TF2. OOP (ala PyTorch), Gradient Tape, Eager Execution      |
| [UK Company Finances, VAE_VQ](https://github.com/olimoz/DataScienceExamples/blob/master/CompaniesHs_Step3_VAE_VQ.py)               	            | Python     | TF2            | VQ  VAE in TF2. OOP (ala PyTorch), Gradient Tape, Eager Execution      |
| [UK Company Finances, GANs, Info+Wasserstein](https://github.com/olimoz/DataScienceExamples/blob/master/CompaniesHs_Step4_GAN_Info_Wasserstein.py)| Python     | TF2            | GANs! Vanilla, InfoGAN and Wasserstein GAN                             |
| [UK Company Finances, LSTMVAE](https://github.com/olimoz/DataScienceExamples/blob/master/CompaniesHs_Step5_LSTMVAE.py)             	            | Python     | TF2            | LSTM AE in TF2. OOP (ala PyTorch), Gradient Tape, Eager Execution      |
| [USA Company Finances, SEC Data Extract](https://github.com/olimoz/DataScienceExamples/blob/master/SEC_Step1_DataExtract.py)             	    | Python     | TF2            | Data extract from US Securities & Exchange Commission                  |
| [USA Company Finances, VAE](https://github.com/olimoz/DataScienceExamples/blob/master/SEC_Step2_VAE.py)             	                            | Python     | TF2            | VAE to create latent space as input to Seq to Seq model                |
| [USA Company Finances, LSTM on Latents](https://github.com/olimoz/DataScienceExamples/blob/master/SEC_Step3_LSTM_on_latents_of_sequential_data.py)| Python     | TF2            | LSTM on Latents. Work ceased, results incomplete.                      |
| [CNN Street Number Data Exploration](https://github.com/olimoz/DataScienceExamples/blob/master/CNN_StreetNumber_DataExploration.pdf)              | R          | N/A            | Working with images in R                                               |
| [CNN Street Number Published Model](https://github.com/olimoz/DataScienceExamples/blob/master/CNN_StreetNumber_PublishedModel.pdf)                | R          | TF1            | Tensorflow 1. 7 layer CNN, classification                              |

