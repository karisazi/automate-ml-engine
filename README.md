# Automate ML Engine: Regression & Forecasting


## 💻 Requirements

### Python version
* Main supported version : <strong>3.8</strong> <br>
* Other supported versions : <strong>3.7</strong> & <strong>3.9</strong>

Please make sure you have one of these versions installed to be able to run the app on your machine.


## ⚙️ Installation


### Create a virtual environment (optional)
We strongly advise to create and activate a new virtual environment, to avoid any dependency issue.

For example with conda:
```bash
pip install conda; conda create -n streamlit_prophet python=3.8; conda activate streamlit_prophet
```

Or with virtualenv:
```bash
pip install virtualenv; python3.7 -m virtualenv streamlit_prophet --python=python3.8; source streamlit_prophet/bin/activate
```


### Install package
Install the needed package:
```bash
pip install -r requirements.txt
```



## 📈 Usage

Once installed, run the following command from CLI to open the app in your default web browser:

<strong>1. Regression</strong>
```bash
cd regression
streamlit run regression.py
```

<strong>2. Forecasting</strong>
```bash
cd forecasting
streamlit run forecasting.py
```