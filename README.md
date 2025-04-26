# Hirefit
# HIRE FIT: Resume-Job Matching Platform

![HIRE FIT Banner](https://img.freepik.com/free-vector/job-hunt-concept-illustration_114360-486.jpg)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  [![Contributors](https://img.shields.io/badge/contributors-4-blue)](https://github.com/)

## Project Overview

**HIRE FIT** is a Big Data and Machine Learning-driven application aimed at automating resume screening and matching the best candidates to job descriptions. By leveraging Hadoop, Hive, Amazon S3, and AWS SageMaker, we efficiently preprocess, store, query, and analyze large volumes of semi-structured data. A binary classification model using XGBoost predicts the job-resume matches based on skill overlap.

---

## Table of Contents
- [Motivation](#motivation)
- [Objectives](#objectives)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Technology Stack](#technology-stack)
- [Workflow](#workflow)
- [Implementation Details](#implementation-details)
- [How to Run](#how-to-run)
- [Results and Visualizations](#results-and-visualizations)
- [Challenges and Solutions](#challenges-and-solutions)
- [Contributors](#contributors)
- [References](#references)

---

## Motivation

In today's fast-paced recruitment environment, finding the right candidate quickly is critical. By preprocessing resumes and job postings using Big Data technologies and applying machine learning models, we can significantly reduce hiring time and improve candidate-job fit.

---

## Objectives

- Preprocess and clean semi-structured resume and job posting datasets.
- Structure the data using Hive and store in Amazon S3.
- Build binary vectors representing skills.
- Train an XGBoost classifier on AWS SageMaker to predict resume-job matches.
- Evaluate and visualize the model's predictions.

---

## Architecture

![Architecture Diagram](https://chat.openai.com/mnt/data/A_diagram_titled_"HIRE_FIT_Architecture"_at_the_to.png)

**(Diagram Flow)**:
- Data Ingestion → Hadoop (Cleaning) → Hive (Structuring) → Amazon S3 (Storage) → AWS SageMaker (Feature Engineering + Model Training) → Predictions

---

## Datasets

- **Source**: Bright Data and Hugging Face.
- **Job Postings Dataset**: Includes company names, job locations, and descriptions.
- **Resumes Dataset**: Semi-structured resumes categorized by job domain.

---

## Technology Stack

- **Big Data Tools**: Hadoop, Hive
- **Cloud Services**: Amazon S3, Amazon SageMaker, AWS EMR
- **Programming & Libraries**: Java (Maven Project), Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Machine Learning**: XGBoost
- **Visualization**: Matplotlib, Seaborn

---

## Workflow

1. **Data Preprocessing**
   - Create Maven projects for both datasets.
   - Run Hadoop MapReduce jobs.
   - Clean, structure, and store the data.

2. **Data Transformation**
   - Create Hive tables.
   - Perform queries: filter, aggregate, and standardize data.

3. **Storage**
   - Save transformed datasets in Amazon S3.

4. **Feature Engineering**
   - Generate binary skill vectors.
   - Calculate Cosine Similarity.

5. **Model Building**
   - Train an XGBoost model on labeled pairs.
   - Optimize thresholds and handle data imbalance.

6. **Evaluation and Visualization**
   - Analyze performance using confusion matrix, scatter plots, and boxplots.

---

## How to Run

> **Prerequisites**:
> - AWS account (S3, EMR, SageMaker access)
> - Python 3.8+
> - Java 8+
> - Maven installed

**1. Preprocessing and Cleaning:**
```bash
# Build Maven project
mvn clean install

# Submit Hadoop jobs
hadoop jar your-preprocessing-jar.jar input_path output_path
```

**2. Structuring Data:**
```sql
-- Create tables in Hive
CREATE TABLE resume (...);
CREATE TABLE job_listings (...);

-- Perform transformations
INSERT INTO standardized_job_listings SELECT ...;
```

**3. Storage:**
- Save Hive outputs to Amazon S3.

**4. Feature Extraction and Modeling:**
```python
# Load data
import pandas as pd
import xgboost as xgb

# Feature engineering
# Train XGBoost classifier
```

**5. Visualization:**
```python
# Generate scatter plots, heatmaps, confusion matrices
```

---

## Results and Visualizations

- Cleaned datasets stored in Amazon S3.
- Cosine Similarity vs Matching Score plots.
- Top 10 matched candidates visualized.
- Scatter plots for Resume vs Job Description Length.
- Confusion Matrix with heatmap.
- Boxplots showing Matching Score Distribution.

---

## Challenges and Solutions

- **Data Preprocessing Complexity**: Maven build and debug steps resolved JAR file issues.
- **Cluster Setup**: Carefully configured EMR clusters.
- **Handling Data Imbalance**: Applied resampling techniques.
- **Semantic Limitations**: Future work includes semantic embeddings.

---

## Contributors

| Name                | Contributions                         |
|---------------------|---------------------------------------|
| Arun Govindgari      | Data Preprocessing, Model Building    |
| Chinmai Kaveti       | Data Preprocessing, Visualizations    |
| Litesh Perumalla     | Data Preprocessing, Visualizations    |
| Pavan Kalyan Natukula| Model Building, Analysis, Documentation|

All members contributed equally to the success of this project.

---

## References

- [Apache Hadoop Documentation](https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)
- [Apache Hive Documentation](https://cwiki.apache.org/confluence/display/Hive/Home)
- [Amazon S3 Documentation](https://docs.aws.amazon.com/s3/index.html)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

> **Thank you for exploring our project! Feel free to star ⭐ the repository if you find it useful!**

![Thank You](https://img.freepik.com/free-vector/thank-you-concept-illustration_114360-7902.jpg)

