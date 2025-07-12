# README: 
### Optional: If Python 3.12 is Not Installed
```
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv
```

### Step 1: Check if Python 3.12 is installed
```
python3.12 --version
```

### Step 2: Create Virtual Environment with Python 3.12
```
python3.12 -m venv wlt-env
```

### Step 3: Activate the Environment
```
source wlt-env/bin/activate
```

### Step 4: Install from requirements.txt
```
pip install -r requirements.txt
```

### Steap 5: Run the Chatbot
```
streamlit run ask_wallet.py
```

### OR:
```
docker-compose up --build
```

### OR fresh built
```
docker-compose build --no-cache                  
docker-compose up
```

### Access the App
```
http://localhost:8501
```

### Docker Compose Down
```
docker-compose down
```

### Extra
```
streamlit run ask_wallet.py
```