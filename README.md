# Textual_Direction_Prediction_Oil_Volatility
Financial markets are influenced by news, investor sentiment, and economic indicators. In this study, we explore whether news alone, without traditional market data, can predict the direction of oil price volatility. Using a decade-long dataset from Eikon (2014–2024), we propose an ensemble learning framework that extracts predictive signals from financial news using sentiment analysis and advanced language models, including GloVe, FastText, FinBERT, BERT, Gemini, and LLaMA.

Our model is benchmarked against the Heterogeneous Autoregressive (HAR) model, and we use the McNemar test for statistical evaluation. While most sentiment indicators underperform HAR, the raw news count emerges as a strong predictor. FastText performs best among text embeddings.

SHAP-based interpretation shows shifting importance of language patterns over time: from supply-demand terms pre-pandemic, to uncertainty during COVID-19, recovery post-shock, and geopolitical focus during war. These results underscore the potential of news-driven NLP features for explainable and robust financial forecasting.
