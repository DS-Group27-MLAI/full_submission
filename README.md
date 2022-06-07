# Code Submitted for Coursework 2 Image Anomaly Detection

## Installation of libraries

**Install Python 3.7 version**

Run the below command:

```bash

pip install -r requirements.txt

```

- Extract the zip file of 01_ML_Project_Code.zip

  - It contains the models copy as well in order to execute the model

- Models are also contained inside 03_Trained_Models_Binary_Files
  
  - |_📄 model_best_weights_classification_1.h5

    - ResNet based Model for **CI7520_Coursework_2_Binary_Classification_ResNet.ipynb**

    - Please rename it to **model_best_weights_classification_resnet_existing_completion.h5** for use with notebook

  - |_📄 model_best_weights_classification_resnet_existing_completion.h5

    - ResNet based Model for **CI7520_Coursework_2_Binary_Classification_ResNet.ipynb**
  
  - |_📄 model_best_weights_classification_densenet_existing_completion.h5

    - DenseNet based Model (Pre-Trained from Keras / Transfer Learning) for **CI7520_Coursework_2_Binary_Classification_DenseNet.ipynb**
  
  - |_📄 model_best_weights_anomaly_vae.h5

    - VAE Model for **CI7520_Coursework_2_Anomaly_Detection_VAE_Existing.ipynb**

    - Please rename it to **model_best_weights_anomaly_detection_vae.h5** for use with the notebook

  - |_📄 vae.tflite

    - The VAE model converted from h5 keras model into 15.1 MB size using **Model Quantization**
  
  - |_📄 model_best_weights_anomaly_detection_vae_designed.h5

    - Conditional VAE Model for **CI7520_Coursework_2_Anomaly_Detection_VAE_Designed.ipynb**
  
  - |_📄 model_best_weights_anomaly_convae.h5

    - Convolutional Autoencoder Model for **CI7520_Coursework2_Anomaly_Detection_ConvAutoEncoder_Designed.ipynb**


## Contents of 01_ML_Project_Code.zip

Shows the contents of the folder:

|_📁 streamlit-app
  - This is the website code that runs streamlit in an app

    |_📁 images
      - Contains sample images

      |_📁 anomaly
        - Sample anomaly images

      |_📁 normal
        - Sample normal images

      |_📁 temp
        - temp folder to load the visualizations dynamically generated
        
    |_📁 models

        |_📁 convae

            |_📄 model_best_weights_anomaly_convae.h5
            - Model copy to execute in streamlit app

        |_📁 convae_code

        |_📁 convae_evaluation

        |_📁 densenet

            |_📄 model_best_weights_classification_densenet_existing_completion.h5
            - Model copy to execute in streamlit app

        |_📁 densenet_code

        |_📁 densenet_evaluation

        |_📁 resnet

            |_📄 model_best_weights_classification_resnet_existing_completion.h5
            - Model copy to execute in streamlit app

        |_📁 resnet_code

            |_📄 y_pred_test.pkl
            - Used by streamlit app

        |_📁 resnet_evaluation

        |_📁 vae

            |_📄 model_best_weights_anomaly_detection_vae.h5
            - Model copy to execute in streamlit app

            |_📄 model_best_weights_anomaly_detection_vae_designed.h5
            - Model copy to execute in streamlit app
            
        |_📁 vae_code
        
            |_📄 anomaly_scores_test.pkl

            |_📄 anomaly_scores_test_noaug.pkl

            |_📄 anomaly_scores_train.pkl

            |_📄 HistGradientBoostingClassifier.pkl

            |_📄 ssim_test_noaug.pkl
            - Files used by streamlit app

        |_📁 vae_evaluation

    |_📄 about.py

    |_📄 anomaly_detection.py

    |_📄 app.py

    |_📄 classification.py

    |_📄 config.py

    |_📄 home.py

    |_📄 inference.py

    |_📄 main.py

    |_📄 manage.py

    |_📄 members.py

    |_📄 multiapp.py

    |_📄 Procfile

    |_📄 README.md

    |_📄 readme.py

    |_📄 requirements.txt

    |_📄 setup.sh

    |_📄 utils.py

    |_📄 visualization.py

|_📄 anomaly_scores_train.pkl

|_📄 CI7520_Coursework2_Anomaly_Detection_ConvAutoEncoder_Designed.ipynb
  - Notebook for Conv AE (Designed DNN) model

|_📄 CI7520_Coursework_2_Anomaly_Detection_VAE_Designed.ipynb
  - Notebook for Conditional VAE (Designed DNN) model

|_📄 CI7520_Coursework_2_Anomaly_Detection_VAE_Existing.ipynb
  - Notebook for VAE (Existing DNN) model

|_📄 CI7520_Coursework_2_Binary_Classification_DenseNet.ipynb
  - Notebook for DenseNet (Existing DNN) model

|_📄 CI7520_Coursework_2_Binary_Classification_ResNet.ipynb
  - Notebook for ResNet (Existing DNN) model

|_📄 HistGradientBoostingClassifier.pkl
  - ML Model load for CVAE Evaluation

|_📄 README.md
  - Read me file

|_📄 requirements.txt
  - requirements file for pip
  
|_📄 y_pred_test.pkl


# StreamLit

## Run streamlit server

```python

cd streamlit-app/

streamlit run app.py --server.port=8501

```

- Navigate to http://localhost:8501

# FastAPI

## Run fastAPI Server

```python

cd streamlit-app/

python main.py

```

- Navigate to http://localhost:8080

