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
### Activate the virtual environment
pipenv shell
### Deactivate when done
exit


pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf python-dotenv
pipenv install huggingface_hub
pipenv install streamlit



