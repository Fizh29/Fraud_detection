FROM python:3.10-slim

# Faster install
RUN pip install --upgrade pip

# Copy project
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install pandas numpy scikit-learn imbalanced-learn \
    openpyxl faker matplotlib seaborn joblib

# Run the entire fraud pipeline
CMD ["python", "main_pipeline.py"]
