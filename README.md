# README: Setting Up Your Environment with Pipenv

## Prerequisite: Install Pipenv
Follow the official Pipenv installation guide to set up Pipenv on your system:  
[Install Pipenv Documentation](https://pipenv.pypa.io/en/latest/installation.html)

---

## Steps to Set Up the Environment

### Install Required Packages
Run the following commands in your terminal (assuming Pipenv is already installed):

```bash
sudo pip3 install pipenv
pipenv install
```
### Activate the virtual environment
```bash
pipenv shell
```
### Deactivate when done
```bash
exit
```

### Packages
```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf python-dotenv huggingface_hub streamlit
pipenv install "numpy>=1.23.5,<2.3.0" sentence-transformers==2.2.2
```

### commands to run
```bash
pipenv run python create_memory_for_llm.py
pipenv run python connect_memory_with_llm.py

streamlit run ask_wallet.py
```


## Docker
### Build the image
```bash
docker build -t ask-wallet-ai .
```
```bash
### Run the container
docker run -p 8501:8501 --env-file .env ask-wallet-ai
```

## Access the App
### open http://localhost:8501




------------------------------------
### Optional: If Python 3.12 is Not Installed
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv

### Step 1: Check if Python 3.12 is installed
python3.12 --version

### Step 2: Create Virtual Environment with Python 3.12
python3.12 -m venv wlt-env

### Step 3: Activate the Environment
source wlt-env/bin/activate

### Step 4: Install from requirements.txt
pip install -r requirements.txt

### Steap 5: Run the Chatbot
streamlit run ask_wallet.py


### OR:
docker-compose up --build

