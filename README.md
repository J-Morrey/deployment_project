# Installation

## Create a conda environment

Run 

```
conda create -n mealplanner python=3.11
conda activate mealplanner
pip install -r requirements.txt
```

## Run the app

Start the backend with 
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

Run frontend with
```
python frontend.py
```