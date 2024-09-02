# Re-examining-the-input-parameters-and-AI-strategies-for-CHF-prediction


### Kai Wang1,2, Da Wang3, Xiaoxing Liu1, Songbai Cheng4, Shixian Wang5*, Wen Zhou5*, Shuichiro Miwa2, Koji Okamoto2
1 Sino-French Institute of Nuclear Engineering and Technology, Sun Yat-Sen University, Zhuhai 519082, China
2 Nuclear Professional School, The University of Tokyo, 2-22 Shirane-shirakata, Tokai-mura, Ibaraki, 319-1188, Japan
3 Iwhalecloud Co., Ltd. Building B, Yihua Industrial Park, Yuhuatai District, Nanjing City
4 Department of Nuclear Engineering and Management, School of Engineering, University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo 113-8654, Japan
5 College of Nuclear Science and Technology, Harbin Engineering University, Harbin 150001, China
*Corresponding Author Email Address: wshixian3@g.ecc.u-tokyo.ac.jp

## Abstract

### Predicting Critical Heat Flux (CHF) remains a crucial and challenging task in two-phase thermohydraulics, as it plays a vital role in preventing heater surface burnout and ensuring system safety. Due to the complex nature of CHF, a universal theory that addresses all related phenomena is yet to be established. As a result, various methods, including empirical and semi-empirical correlations, look-up tables (LUT), and Computational Fluid Dynamics (CFD) approaches, have been developed for CHF prediction. However, traditional methods often suffer from issues related to accuracy and generalizability. Recent advancements in Artificial Intelligence (AI) have demonstrated significant potential in improving CHF prediction accuracy. In this study, we employ Transformer, Mamba, and Temporal Convolutional Network (TCN) models to predict CHF, with Transformers outperforming the other methods, thereby solidifying their leading position. Our research re-examines the input parameters used in previous studies, which often relied on indirect thermohydraulic parameters, reducing prediction accuracy. By carefully selecting input parameters based on mechanistic analyses and employing Transformer models, we achieved a minimum Normalized Root Mean Square Error (NRMSE) of 6.63% using experimental data exceeding 20,000 points—the lowest reported to date. Additionally, we tested five traditional AI methods: Random Forest, Ridge Regression, Bayesian Linear Regression, ε-SVM, and Backpropagation Neural Network (BPNN). While most traditional methods underperformed compared to LUT, the Random Forest model achieved an NRMSE of 4.39%, highlighting its potential for accurate CHF prediction. This study provides guidance for future AI-based CHF prediction efforts in two-phase thermohydraulic applications.

### Keywords: Critical heat flux; Artificial Intelligence; Deep learning; Neural network; In-put parameters
