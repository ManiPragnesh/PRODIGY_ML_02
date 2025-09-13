# Customer Segmentation using K-Means Clustering

## Project Objective
This project aims to segment customers based on their purchasing behavior using the K-Means clustering algorithm. By analyzing customer data, we can identify distinct groups with similar characteristics, enabling targeted marketing strategies and personalized customer experiences.

## Dataset Description
The dataset used is `Mall_Customers.csv` from Kaggle, containing the following key columns:
- **CustomerID**: Unique identifier for each customer
- **Gender**: Customer's gender (Male/Female)
- **Age**: Customer's age
- **Annual Income (k$)**: Annual income in thousands of dollars
- **Spending Score (1-100)**: A score representing the customer's spending behavior (higher values indicate higher spending)

## Workflow Steps
1. **Data Preprocessing**:
   - Load and inspect the dataset.
   - Handle missing values (if any).
   - Standardize numerical features (e.g., `Annual Income` and `Spending Score`) for clustering.

2. **Feature Selection**:
   - Select relevant features for clustering (`Annual Income` and `Spending Score`).

3. **Model Training**:
   - Train the K-Means clustering algorithm on the selected features.
   - Use the elbow method to determine the optimal number of clusters (`K`).

4. **Clustering Process**:
   - Assign customers to clusters based on their features.
   - Visualize the clusters using scatter plots.

5. **Evaluation**:
   - Analyze the characteristics of each cluster (e.g., average income, spending score).
   - Interpret the clusters to derive actionable insights.

## Results and Insights
- **Cluster Characteristics**: Each cluster represents a distinct customer segment with unique spending and income patterns.
- **Actionable Insights**: For example, high-income, high-spending customers can be targeted with premium offers, while low-income, high-spending customers may benefit from loyalty programs.

## Technical Stack
- **Programming Language**: Python
- **Libraries**:
  - `pandas` for data manipulation
  - `scikit-learn` for K-Means clustering
  - `matplotlib` and `seaborn` for visualizations
  - `streamlit` for the interactive web application (optional)
- **Tools**: Jupyter Notebook (for exploration), Docker (for deployment)

## How to Run the Project
1. **Installation**:
   - Ensure Python 3.9+ is installed.
   - Install the required libraries using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Running the Script**:
   - For the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - For Docker deployment:
     ```bash
     docker build -t customer-segmentation-app .
     docker run -p 8501:8501 customer-segmentation-app
     ```

3. **Access the Application**:
   - Open `http://localhost:8501` in your browser to interact with the Streamlit app.

## Notes
- The trained model and visualizations are saved in the `assets/` directory.
- For detailed code and implementation, refer to the Jupyter notebook (`notebooks/customer_segmentation_kmeans.ipynb`) or the Python script (`app.py`).