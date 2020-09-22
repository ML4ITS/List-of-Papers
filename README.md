# ML4ITS - Machine Learning for Irregular Time Series

Repository with code and resources related to the current activity on time series analysis.

## Interesting papers/resources
Refer to [this](https://docs.google.com/document/d/1qw6QkqPOySN6NbkHx6qfSaWEWPHeNW0y7Mgx2uccbH0/edit?usp=sharing) document for a curated list of papers/articles.

[Here](https://docs.google.com/document/d/1xtFhYKeJX-qfHBGx1T-zCZIrQ2DmCI0ZgpDcq5O8X0w/edit?usp=sharing) some updated notes on articles/papers read

### Time Series Classification
- *Deep Learning for Time Series Classification (InceptionTime)* (2020) [[post]](https://towardsdatascience.com/deep-learning-for-time-series-classification-inceptiontime-245703f422db)
- *Deep learning for time series classification: a review* (DMKD2019) [[paper]](https://dl.acm.org/doi/10.1007/s10618-019-00619-1)

### Time Series Forecasting
- *Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting* (NIPS2019) [[paper]](http://papers.neurips.cc/paper/8730-think-globally-act-locally-a-deep-neural-network-approach-to-high-dimensional-time-series-forecasting)
- *Multivariate Temporal Convolutional Network:A Deep Neural Networks Approach for Multivariate Time Series Forecasting* (MDPI2019) [[paper]](https://www.mdpi.com/2079-9292/8/8/876)
- *Joint Modeling of Local and Global Temporal Dynamics for Multivariate Time Series Forecasting with Missing Values* (IRREGULAR, forecasting with Missing values AAAI2020)
- *Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models* (IRREGULAR, Non-stationary - NIPS2020)
- *DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING* (ICLR2018) [[paper]](https://openreview.net/pdf?id=SJiHXGWAZ)
- *Time-series Extreme Event Forecasting with Neural Networks at Uber* (2017) (Autoencoder and LSTM for rare event prediction in univariate TS)


### Anomaly Detection / Failure Prediction
- *Generative Adversarial Networks for Failure Prediction* (ECML2019) [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-46133-1_37)
- *A GAN-Based Anomaly Detection Approach for Imbalanced Industrial Time Series* (IEEE2019) [[paper]](https://ieeexplore.ieee.org/document/8853246)
- *DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series* (IEEE2019) [[paper]](https://ieeexplore.ieee.org/document/8581424)
- *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network* (KDD2019)

### GAN for Data Imputation (Time Series Domain and not)
- [NoTS] *GAIN: Missing Data Imputation using Generative Adversarial Nets* (IJCAI2018) [[paper](http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf)] [[code (tensorflow)](https://github.com/jsyoon0823/GAIN), [code2](https://github.com/lethaiq/GAIN)] [mycode]
- [MTS] *E²GAN: End-to-End Generative Adversarial Network for Multivariate Time Series Imputation* (IJCAI2019) [[paper](https://www.ijcai.org/Proceedings/2019/429)] [[code (tensorflow)](https://github.com/Luoyonghong/E2EGAN)] [mycode]
- [NoTS] *MisGAN: Learning from Incomplete Data with Generative Adversarial Networks* (ICLR2019) [[paper](https://arxiv.org/abs/1902.09599)] [[code (tensorflow)](https://github.com/steveli/misgan)] [mycode]
- [GAN and Missing Data Imputation (medium post)](https://towardsdatascience.com/gans-and-missing-data-imputation-815a0cbc4ece)
- *Recurrent Neural Networks for Multivariate Time Series with Missing Values* (Nature 2018) [[paper]](https://www.nature.com/articles/s41598-018-24271-9)

### Generative Models for (Multivariate) Time Series
- *Quant GANs: Deep Generation of Financial Time Series* (arXiv2019) [[paper]](https://arxiv.org/abs/1907.06673)
- *Generating Financial Series with Generative Adversarial Networks (blog post)* [[part1]](https://quantdare.com/generating-financial-series-with-generative-adversarial-networks/)[[part2]](https://quantdare.com/generating-financial-series-with-gans-ii/)
- *Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs* (arXiv2017) [[paper]](https://arxiv.org/abs/1706.02633) [[code (pytorch)]](https://github.com/proceduralia/pytorch-GAN-timeseries)

### Attention Mechanism for Time Series Analysis
- *Forecasting stock prices with long-short term memory neural network based on attention mechanism* (PlosONE 2020)
- *DATA-GRU: Dual-Attention Time-Aware Gated Recurrent Unit for Irregular Multivariate Time Series* (IRREGULAR, AAAI2020)
- *DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting* (CIKM2019 - Short)
- *Multivariate Time Series Early Classification with Interpretability Using Deep Learning and Attention Mechanism (ECML-PKDD2019)* [TSC,MV]
- *Modeling Extreme Events in Time Series Prediction* (KDD2019) [TSF, UV]
- *Multi-Horizon Time Series Forecasting with Temporal Attention Learning* (KDD2019)
- *CAMP: Co-Attention Memory Networks for Diagnosis Prediction in Healthcare* (ICDM2019)
- *Temporal pattern attention for multivariate time series forecasting* (ECML-PKDD2019) [TSF,MV]
- *MuVAN: A Multi-view Attention Network for Multivariate Temporal Data* (ICDM2018)
- *Attend and Diagnose: Clinical Time Series Analysis Using Attention Models* (AAAI2018)
- *A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction* (IJCAI2017) [TSF, UV]

### Transfer Learning for Time Series
- *ConvTimeNet: A Pre-trained Deep Convolutional Neural Network for Time Series Classification* (IJCNN2019) [TSC, UV]
- *Transfer Learning for Financial Time Series Forecasting* (PRICAI2019) [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-29911-8_3)
- *Time Series Anomaly Detection Using Convolutional Neural Networks and Transfer Learning* (arXiv2019) [[paper]](https://arxiv.org/abs/1905.13628)
- *Multi-source transfer learning of time series in cyclical manufacturing* (Journal of Intelligent Manufacturing 2019) [[paper]](https://link.springer.com/article/10.1007/s10845-019-01499-4)
- *Transfer Learning Based Fault Diagnosis with Missing Data Due to Multi-Rate Sampling* (MDPI2019) [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6514833/)
- *Transfer learning for time series classification* (IEEE Conference on Big Data2018) [[paper]](https://ieeexplore.ieee.org/document/8621990)
- *Reconstruction and Regression Loss for Time-Series Transfer Learning* (SIGKDD MiLeTS' 2018) [[paper]](https://milets18.github.io/papers/milets18_paper_2.pdf)
- *Towards a universal neural network encoder for time series* (CCIA2018) [TSC, UV] 
- *Transfer Learning with Deep Convolutional Neural Network for SAR Target Classification with Limited Labeled Data* (MDPI2017) [[paper]](https://www.mdpi.com/2072-4292/9/9/907)

### Unsupervised Learning, Representation Learning and Self Supervised Learning for TS
- *Self-Supervised Learning for Semi-Supervised Time Series Classification* (PAKDD2020) [TSC, MV]
- *Self-supervised representation learning from electroencephalography signals* (IEEE MLSP 2019)
- *Unsupervised Scalable Representation Learning for Multivariate Time Series* (NIPS2019) [[paper]](https://papers.nips.cc/paper/8713-unsupervised-scalable-representation-learning-for-multivariate-time-series) [[code]](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries)
- *Deep Multivariate Time Series Embedding Clustering via Attentive-Gated Autoencoder* (PAKDD2020) [TSClusering, MV]
- *Learning Representations for Time Series Clustering* [TSClustering, UV] (NIPS2019)
- *Unsupervised Scalable Representation Learning for Multivariate Time Series* [TSC, MV] (NIPS2019)
- *Unsupervised pre-training of a Deep LStM-based Stacked Autoencoder for Multivariate time Series forecasting problems* (NatureSR 2019) [TSF,MV]
- *Adversarial Unsupervised Representation Learning for Activity Time-Series* (AAAI2019)
- *A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data* (AAAI2019) [TSAD,MV]
- *TimeNet: Pre-trained deep recurrent neural network for time series classification*
- *Unsupervised Pre-training of a Deep LSTM-based Stacked Autoencoder for Multivariate Time Series Forecasting Problems* (Nature2019) [TSF,MV]

### Few-Shot Learning for Time Series Classification in Low-Data Regime 
- *TapNet: Multivariate Time Series Classification with Attentional Prototypical Network* [TSC, MV] (AAAI2020)
- *Meta-Learning for Few-Shot Time Series Classification* (2019) [[paper]](https://arxiv.org/abs/1909.07155) (ACM-IKDD2020) [UV,TSC]
- *Deep Prototypical Networks for Imbalanced Time Series Classification under Data Scarcity* (CIKM2019 - Short)
- *Few-shot Time-series Classification with Dual Interpretability* [TSC]

### Reservoir Computing for TS Analysis
- *Analysis of Wide and Deep Echo State Networks for Multiscale Spatiotemporal Time Series Forecasting* (NICE2019) [[paper]](https://dl.acm.org/doi/10.1145/3320288.3320303)
- *Network Traffic Prediction Using Variational Mode Decomposition and Multi- Reservoirs Echo State Network* (IEEE2019) [[paper]](https://ieeexplore.ieee.org/abstract/document/8846010)
- *Spiking Echo State Convolutional Neural Network for Robust Time Series Classification* (IEEE2018) [[paper]](https://ieeexplore.ieee.org/document/8580574)

### Time Series and Asynchronous data
- Modeling asynchronous event sequences with RNNs [[paper]](https://www.sciencedirect.com/science/article/pii/S1532046418300996)

### Latent Models and ODE models for Time Series Modeling
 - *Neural ODEs* (2019) [[paper]](https://arxiv.org/abs/1806.07366), [[code (pytorch)]](https://github.com/rtqichen/torchdiffeq)
 - *Latent ODEs for Irregularly-Sampled Time Series* [[paper]](https://arxiv.org/abs/1907.03907), [[code (pytorch)]](https://github.com/YuliaRubanova/latent_ode)

### Time Series as Images
- *Multivariate Time Series as Images: Imputation Using Convolutional Denoising Autoencoder (2020)* [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-44584-3_1)

### Pytorch Packages
- Pytorch Forecasting [Library](https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46) [blogpost](https://pytorch-forecasting.readthedocs.io/en/latest/)
- pyts: a Python package for time series classification [github](https://github.com/johannfaouzi/pyts)

## Dataset
- [25 Datasets for Deep Learning in IoT](https://hub.packtpub.com/25-datasets-deep-learning-iot/?utm_source=affiliate&utm_medium=rakuten&utm_campaign=2126220:adgoal.net&utm_content=10&utm_term=us_network&ranMID=45060&ranEAID=a1LgFw09t88&ranSiteID=a1LgFw09t88-rGnxSa5HZICOQ9ewmmK8Kg)
- [41 Multivariate Time Series from UCI Repository](https://archive.ics.uci.edu/ml/datasets.php?format=&task=&att=&area=&numAtt=10to100&numIns=&type=ts&sort=nameUp&view=table)
- [UEA & UCR Time Series Classification Repository](http://www.timeseriesclassification.com/index.php) [[paper]](https://arxiv.org/abs/1811.00075)
- [SmartMeter Energy Consumption Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
- [NVE Hydrological API (HydAPI)](https://hydapi.nve.no/UserDocumentation/)
- [Wikipedia Web Traffic Time Series](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
- [Measuring Broadband America (MBA)](https://www.fcc.gov/reports-research/reports/measuring-broadband-america/raw-data-measuring-broadband-america-seventh)
- [Google Cluster Usage Traces (GCUT)](https://github.com/google/cluster-data)
- [Physionet MIMIC-III](https://mimic.physionet.org/)

### Blog Post and other resources
- List of state of the art papers focus on deep learning and resources !!!!! [[github repo]](https://github.com/Alro10/deep-learning-time-series)
- On the Automation of Time Series Forecasting Models: Technical and Organizational Considerations. [[blogpost]](https://towardsdatascience.com/on-the-automation-of-time-series-forecasting-models-technical-and-organizational-considerations-286db3120c8e)
- PyTorch Dataset for multivariate time series [[code]](https://github.com/JulesBelveze/time-series-dataset)


## Examples
- [Example-1: CNN for univariate TS forecasting](./Example-1/)
- [Example-2: Simple GAN for learning and generate a simple math function](./Example-2/)
