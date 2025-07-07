

# 📦 Supply Chain Business Model - Product Weight Prediction

This is an end-to-end machine learning project focused on predicting the **product weight in tons (`product_wg_ton`)** in a supply chain setup. By using data preprocessing, outlier handling, dimensionality reduction (PCA), and regression modeling (XGBoost), we aim to build a reliable model for accurate product weight estimation.

# What is a Supply Chain?
A supply chain is the complete process of making and delivering a product — from raw materials to the final product reaching the customer. It includes all the steps involved in producing, storing, transporting, and selling goods.

# Key Components of a Supply Chain:
Suppliers:
Provide raw materials or components needed to make the product.

Manufacturing:
The process of converting raw materials into finished goods.

Warehousing:
Storage of raw materials and finished products before they are sold or shipped.

Distribution:
Moving the products from factories or warehouses to wholesalers, retailers, or customers.

Retailers/Customers:
Final delivery of products to end-users or consumers.

Information Flow:
Sharing data across all stages (e.g., order details, inventory levels) to make better decisions.

Logistics:
The coordination of transportation, storage, and delivery to ensure goods move efficiently.

## 🧾 Table of Contents

- [Objective](#objective)
- [Dataset Overview](#dataset)
- [Technologies Used](#technologies)
- [Workflow Steps](#workflow)
- [Model Performance](#performance)
- [How to Run](#run)
- [Future Improvements](#future)
- [OOP Concepts](#oops)
- [Author](#author)

---

## 🎯 Objective <a name="objective"></a>

To develop a regression model that accurately predicts **product weight in tons** (`product_wg_ton`) using features related to warehouse operations, capacity, and logistics in the supply chain. The final model should help in operational forecasting and logistics optimization.

To accurately predict the product weight (in tons) using machine learning regression techniques.

To analyze key features that impact product weight and improve model performance.

To develop a robust model that can help in efficient inventory management and logistics planning.

To minimize prediction errors and optimize resource allocation based on weight forecasts.

Sure! Here’s a simpler version of the problem statement:

---

### Problem Statement:

It is hard to guess the exact weight of products by hand, which can cause mistakes in shipping and inventory. This project aims to create a machine learning model that can quickly and accurately predict product weight using available data, helping to save time and reduce errors.

## 📂 Dataset Overview <a name="dataset"></a>

- **Source:** [SCM.csv on GitHub](https://raw.githubusercontent.com/MontyVasita18/SupplyChain_BusinessModel/refs/heads/main/SCM.csv)
- **Target Column:** `product_wg_ton` (Product weight in tons)
- **Features:**
  - Warehouse ID, Manager ID, Workers
  - Capacity, Region, Zone, Location Type
  - Owner Type, Certification Status
  - and more...

---

## 🧰 Technologies Used <a name="technologies"></a>

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation
  - `seaborn`, `matplotlib` – Data visualization
  - `scikit-learn` – Preprocessing, PCA, modeling
  - `xgboost` – Regression
  - `logging`, `warnings` – Runtime handling

---

## ⚙️ Workflow Steps <a name="workflow"></a>

### 1. Data Cleaning
- Dropped unnecessary columns
- Filled missing values (e.g., median for `workers_num`, 'C' for certifications)

### 2. Encoding & Outlier Handling
- Used `LabelEncoder` for categorical variables
- Replaced numeric outliers using IQR method and median

### 3. Dimensionality Reduction
- Applied **MinMaxScaler** for normalization
- Used **PCA** to retain ≥90% variance (reduced dimensions)

### 4. Model Building
- Split data into train and test sets (70/30)
- Trained **XGBoost Regressor**
- Evaluated model using **R² score**

---

## 📈 Model Performance <a name="performance"></a>

| Metric   | Value   |
|----------|---------|
| R² Score (Training) | **~0.95** |
| Model Type | XGBoost Regressor |
| Problem Type | Regression |

> ✅ The model captures ~95% of the variance in training data, showing strong predictive capability.

---

## ▶️ How to Run the Project <a name="run"></a>

1. Clone the repository:
   ```bash
   git clone https://github.com/MontyVasita18/SupplyChain_BusinessModel.git
   cd SupplyChain_BusinessModel
`

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:

   ```bash
   jupyter notebook model.ipynb
   ```

---

## 🚀 Future Improvements <a name="future"></a>

* Evaluate model on test data and compare metrics
* Add hyperparameter tuning using `GridSearchCV`
* Try alternative models (e.g., Random Forest, SVR)
* Deploy with Flask or Streamlit for real-time use
* Integrate SHAP for explainable AI

---

## 🧠 OOPs Concepts Used <a name="oops"></a>

This project leverages OOPs (Object-Oriented Programming) in Python:

| Concept        | Description                                    |
| -------------- | ---------------------------------------------- |
| Class & Object | Blueprint and instance                         |
| Inheritance    | Reuse code from parent class                   |
| Polymorphism   | Same method, different behavior                |
| Encapsulation  | Hiding data and providing access via methods   |
| Abstraction    | Hiding complex logic, exposing only essentials |

Example usage includes model class abstraction, scikit-learn's object-based structure, and structured data pipeline.


## 👤 Author <a name="author"></a>

**Monty Kishan Vasita**
📧 Email: \[[montyvasita1@.com](mailto:your_email@example.com)]
🌐 GitHub: [MontyVasita18](https://github.com/MontyVasita18)


## 🏷️ Tags

`#Regression` `#SupplyChain` `#PCA` `#XGBoost` `#Python` `#DataScience` `#OOPs`





