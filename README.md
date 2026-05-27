Install:
cd backend
venv\Scripts\activate
npm install,
pip install flask flask-cors python-dotenv 
pip install scikit-learn pandas numpy scipy spacy datasets
python -m spacy download en_core_web_sm
or
pip install -r requirements.txt  


To run backend use commands:
cd .\backend\   
python -m venv venv  
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  (optional if getting error)
venv\Scripts\activate   
python train_models.py
python app.py     

To run frontend :Directly click on "Open Live Server"
