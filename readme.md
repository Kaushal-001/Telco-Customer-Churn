
## 🧩 Project Description – Telco Customer Churn Prediction

The **Telco Customer Churn Prediction** project is a complete **end-to-end machine learning solution** designed to help telecom companies identify customers who are likely to discontinue their services. By leveraging historical customer data, the model predicts churn probability based on service usage, payment patterns, and demographic information.

This project not only focuses on building a robust prediction model but also demonstrates the **deployment workflow** through **FastAPI** and **Docker**, making it production-ready and easy to integrate into real-world systems.

---

### 🎯 Objectives

- Analyze customer data to understand patterns influencing churn.  
- Build and train a machine learning model to predict churn.  
- Develop an API for real-time predictions using **FastAPI**.  
- Containerize the application with **Docker** for scalable deployment.  
- Provide a simple and interactive **UI** to visualize predictions.  

---

### ⚙️ Workflow

**1. Data Preprocessing:**  
Cleaning, encoding categorical variables, scaling numerical data, and handling missing values.  

**2. Model Building:**  
Trained multiple models including **Logistic Regression**, **Random Forest**, and **XGBoost** to find the best-performing one.  

**3. Model Evaluation:**  
Evaluated models based on metrics such as **Accuracy**, **Precision**, **Recall**, and **ROC-AUC Score**.  

**4. API Development:**  
Created REST endpoints using **FastAPI** to handle prediction requests.  

**5. Containerization:**  
Used **Docker** to build and run the application in isolated environments for consistent deployment.  

**6. Visualization & UI:**  
Built an intuitive interface for users to input customer details and view prediction results interactively.  

---

### 🚀 Key Features

- End-to-end **ML pipeline** from data ingestion to deployment.  
- **FastAPI backend** for real-time inference.  
- **Dockerized architecture** for easy portability and deployment.  
- **Visual analytics** to understand churn behavior and model performance.  
- Ready for **integration with CI/CD and cloud platforms** (future scalability).  


## 🛠 Tech Stack Used

### 💻 Programming & Database
- **Languages:** Python  

---

### 🧠 Machine Learning & Data Science
- **Libraries:** Scikit-learn | Pandas | NumPy  
- **Visualization:** Matplotlib | Seaborn  

---

### 🐳 Deployment & DevOps
- **Tools:** Docker | Git | GitHub Actions | FastAPI | Streamlit  | MLFlow

---

### 🧰 Development 
- **IDEs & Tools:** VS Code | Jupyter Notebook


<h3>🖥 Project UI Preview</h3> <p align="center"> <img src="https://github.com/Kaushal-001/Telco-Customer-Churn/blob/main/0.0.0.0_8000_ui_.png" alt="Telco Customer Churn Project UI" width="700"> </p>

---

<h3>🚀 Run Locally</h3>

```bash
# Clone this repository
git clone https://github.com/Kaushal-001/Telco-Customer-Churn.git

# Navigate to the folder
cd Telco-Customer-Churn

# Build Docker image
docker build -t telco-customer .

# Run the container
docker run -p 8000:8000 telco-customer
