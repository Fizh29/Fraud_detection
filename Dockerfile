 # seaborn \
    # joblib \
    # google-generativeai \
# numpy \
    # scikit-learn \
    # imbalanced-learn \
    # openpyxl \
    # faker \
FROM python:3.10-slim

# Faster install
RUN pip install --upgrade pip

# Copy project
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install \
    pandas \
    matplotlib \
    streamlit

# Default command
EXPOSE 8501

CMD ["python", "main_pipeline.py"]
