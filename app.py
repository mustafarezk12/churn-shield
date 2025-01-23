import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model and mean/std values
model_filename = 'model(1).pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open('mean_std_values.pkl', 'rb') as f:
    mean_std_values = pickle.load(f)
st.set_page_config(page_title="Customer Churn", layout="wide")

# 1. Define the Home page
def home_page():
    outer_col1, outer_col2, outer_col3 = st.columns([1, 5, 1])
    with outer_col2:
        with st.container():
            col1, col2 = st.columns([2.5, 1])
            with col1:
                st.header("About this app", anchor=None)
                st.markdown(
                    "<div style='font-size:22px;'>- Easily predict if a customer is likely to churn or not using our <span style='color:red;'>Customer Churn Predictor</span>.<br>"
                    "- View customer churn behavior using our <span style='color:red;'>Insights</span>.</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.image("Header - Voluntary vs.png", width=400)

        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.header("What is customer churn?")
                st.write(
                    "<div style='font-size:20px;'>Customer churn occurs when customers stop using a company's products. This can result from various factors. "
                    "Customer features like ratings and usage metrics provide insight into customer behavior, especially when they're about to churn.</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.header("Why predict customer churn?")
                st.write(
                    "<div style='font-size:20px;'>It's much more expensive to acquire new customers than to retain existing ones. "
                    "Predicting customer churn and identifying early warning signs can save significant costs for a company.</div>",
                    unsafe_allow_html=True
                )

        
        st.markdown("<hr style='border: 2px solid gray;'>", unsafe_allow_html=True)

        
        st.markdown("<h2 style='text-align: center;'>About Dataset</h2>", unsafe_allow_html=True)

        st.write(
            "<div style='font-size:18px; text-align: justify;padding: 20px;'>The churn label indicates whether a customer has churned or not. A churned customer is one who has decided to discontinue their subscription or usage of the company's services. On the other hand, a non-churned customer is one who continues to remain engaged and retains their relationship with the company.</div>",
            unsafe_allow_html=True
        )
        st.write(
            "<div style='font-size:18px; text-align: justify;padding: 20px;'>The dataset includes customer information such as age, gender, tenure, usage frequency, support calls, payment delay, "
            "subscription type, contract length, total spend, and last interaction details. This information is used to predict whether a customer is likely to churn based on these behaviors.</div>",
            unsafe_allow_html=True
        )
        st.write(
    "<div style='font-size:18px; text-align: justify; padding: 20px;'>"
    "These datasets contain 12 feature columns. In detail, these are:<br><br>"
    "<ul>"
    "<li><b>CustomerID:</b> A unique identifier for each customer</li>"
    "<li><b>Age:</b> The age of the customer</li>"
    "<li><b>Gender:</b> Gender of the customer</li>"
    "<li><b>Tenure:</b> Duration in months for which a customer has been using the company's products or services</li>"
    "<li><b>Usage Frequency:</b> Number of times the customer has used the company’s services in the last month</li>"
    "<li><b>Support Calls:</b> Number of calls the customer has made to customer support in the last month</li>"
    "<li><b>Payment Delay:</b> Number of days the customer has delayed their payment in the last month</li>"
    "<li><b>Subscription Type:</b> Type of subscription the customer has chosen</li>"
    "<li><b>Contract Length:</b> Duration of the contract the customer has signed with the company</li>"
    "<li><b>Total Spend:</b> Total amount of money the customer has spent on the company's products or services</li>"
    "<li><b>Last Interaction:</b> Number of days since the last interaction the customer had with the company</li>"
    "<li><b>Churn:</b> Binary label indicating whether a customer has churned (1) or not (0)</li>"
    "</ul>"
    "</div>",
    unsafe_allow_html=True
)

# 2. Define the Prediction page
def predict_page():
    outer_col1, outer_col2, outer_col3 = st.columns([1, 5, 1])
    with outer_col2:
        # Add image centered
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.image("image.png", use_column_width=True)

        st.title('Customer Behavior Prediction')

        # Input fields for the user to enter data
        st.header("Enter Customer Details:")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        usage_frequency = st.number_input("Usage Frequency (Last Month)", min_value=0, max_value=100, value=10)
        support_calls = st.number_input("Support Calls (Last Month)", min_value=0, max_value=50, value=3)
        payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=100, value=5)
        subscription_type = st.selectbox("Subscription Type", ["Standard", "Premium", "Basic"])
        contract_length = st.selectbox("Contract Length", ["Monthly", "Annual", "Quarterly"])
        total_spend = st.number_input("Total Spend", min_value=0.0, value=500.0, step=0.1)
        last_interaction = st.number_input("Last Interaction (Days)", min_value=0, max_value=100, value=20)

        # Convert categorical features to numerical
        gender_dict = {"Male": 1, "Female": 0}
        subscription_dict = {"Standard": 2, "Premium": 1, "Basic": 0}
        contract_dict = {"Monthly": 1, "Annual": 0, "Quarterly": 2}

        gender_val = gender_dict[gender]
        subscription_val = subscription_dict[subscription_type]
        contract_val = contract_dict[contract_length]

        # Prepare the input data
        input_data = np.array([age, gender_val, tenure, usage_frequency, support_calls, 
                               payment_delay, subscription_val, contract_val, 
                               total_spend, last_interaction]).reshape(1, -1)

        # Normalize the input data
        input_data = (input_data - mean_std_values['mean']) / mean_std_values['std']

        # Add the predict button
        if st.button('Predict'):
            # Perform the prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            if prediction[0] == 1:
                bg_color = 'red'
                prediction_result = 'The customer has churned'
            else:
                bg_color = 'green'
                prediction_result = 'The customer has not churned'

            confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

            # Display the result
            st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:15px; font-size:18px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

# Visualization functions

def make_histogram(df, target_feature, bins=10, custom_ticks=None, unit='', additional=''):
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure and axis explicitly
    ax.hist(df[target_feature], bins=bins)
    
    if custom_ticks is not None:
        ax.set_xticks(custom_ticks)
        
    ax.set_ylabel('Count')
    ax.set_xlabel(target_feature)
    ax.set_title(f"Distribution of {target_feature.lower()}{additional}:\n")
    ax.grid()
    
    return fig  # Return the figure object

def make_piechart(df, target_feature, additional=''):
    dict_of_val_counts = dict(df[target_feature].value_counts())
    data = list(dict_of_val_counts.values())
    keys = list(dict_of_val_counts.keys())
    
    fig, ax = plt.subplots(figsize=(2, 2))  
    palette_color = sns.color_palette('Set2')  

    
    ax.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%', 
           textprops={'fontsize': 8})  
    ax.set_title(f"Distribution of Customer's {target_feature}", fontsize=10)  

    return fig  # Return the figure object


def make_barplot(df, target_feature, custom_ticks=None, unit='', additional=''):
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure and axis explicitly
    dict_of_val_counts = dict(df[target_feature].value_counts())
    data = list(dict_of_val_counts.values())
    keys = list(dict_of_val_counts.keys())
    
    ax.bar(keys, data)
    
    if custom_ticks is not None:
        ax.set_xticks(custom_ticks)
        
    ax.set_xlabel(f'{target_feature.capitalize()}{additional}')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Distribution of Customer's {target_feature.lower()}{additional}\n")
    ax.grid(axis='y')
    
    return fig  # Return the figure object

def make_boxplot(df, feature):
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure and axis explicitly
    sns.boxplot(df[feature], ax=ax)
    
    ax.set_title(f"Boxplot of {feature}\n")
    ax.set_xlabel(feature)
    ax.set_ylabel("Values")
    
    return fig  # Return the figure object

# Define the Insights page in Streamlit
def insights_page():
    outer_col1, outer_col2, outer_col3 = st.columns([1, 5, 1])
    with outer_col2:
        with st.container():
            st.header("Customer Churn Behavior Analysis")
            st.write("In this section, you can view visual insights regarding customer churn behavior.")
            
            # Load the dataset
            df = pd.read_csv('customer_churn.csv')

            # Display the first few rows of the dataset for validation
            st.subheader("Dataset Preview")
            st.write(df.head())

            # Add visualizations
            st.markdown("<hr style='border: 2px solid gray;'>", unsafe_allow_html=True)
            
            # Visualization 1: Pie Chart - Customer's Gender
            fig1 = make_piechart(df, target_feature='Gender')  # Get the figure from the function
            st.pyplot(fig1)  # Pass the figure object
            
            # Explanation for the plot
            st.write("""
            **Note:** There are more male customers in the company.
            """)
            
            # Visualization 2: Pie Chart - Subscription Type
            
            fig2 = make_piechart(df, target_feature='Subscription Type')  # Get the figure from the function
            st.pyplot(fig2)  # Pass the figure object

            # Explanation for the plot
            st.write("""
            **NOte:** There is a close balance of customers among the three subscription types: Standard, Premium, and Basic.
            """)
            
             # Visualization 2: Pie Chart - Subscription Type
            
            fig3 = make_piechart(df, target_feature='Contract Length')  # Get the figure from the function
            st.pyplot(fig3)  # Pass the figure object

            # Explanation for the plot
            st.write("""
            **Note:** Annual contracts and quarterly contracts have similar and the highest number of customers counts, followed by monthly contracts with the lowest customers
            """)
            
            fig4 = make_barplot(df, target_feature='Age',custom_ticks=np.arange(0, 66, 5), additional=' (years)', unit='years')  # Get the figure from the function
            st.pyplot(fig4)  # Pass the figure object

            # Explanation for the plot
            st.write("""
            **Note:** Most customers are aged 40-50 with age 50 being the most common. There's very low number of customers of age 51 and above.
            """)
            fig5 = make_barplot(df, target_feature='Support Calls', unit='calls', additional=' (in a month)')  # Get the figure from the function
            st.pyplot(fig5)  # Pass the figure object

            # Explanation for the plot
            st.write("""
            **Note:** On average, customers tend to make 3 support calls in a month. Customers tend to make 1 or 2 support calls per month, with the most make no support calls at all.
            """)
            
            # Gender-wise churn rate visualization
            gender_churn = df.groupby(['Gender', 'Churn']).size().unstack()

            X = list(gender_churn.index)
            churn_0 = list(gender_churn.iloc[:, 0])
            churn_1 = list(gender_churn.iloc[:, 1])

            X_axis = np.arange(len(X))

            # Create the bar chart
            fig_churn = plt.figure(figsize=(10, 5))
            plt.bar(X_axis - 0.2, churn_1, 0.4, label='Churn')
            plt.bar(X_axis + 0.2, churn_0, 0.4, label='Not Churn')

            plt.xticks(X_axis, X)
            plt.xlabel('Gender')
            plt.ylabel('Count')
            plt.title("Gender-wise Churn Rate")
            plt.legend(loc='center right')
            plt.grid(axis='y')

            # Gender-wise churn rate bar chart          
            st.pyplot(fig_churn)
            
            st.write("""
            **Note:** Gender and churn rate have relationship.

Female customers exhibit a slightly higher churn rate compared to male customers. Active male customers (non-churned) is nearly double that of female customers.
            """)
            
            filtered = df.groupby(['Payment Delay', 'Churn']).size().unstack()

            X = list(filtered.index)
            churn_0_delay = list(filtered.iloc[:, 0])
            churn_1_delay = list(filtered.iloc[:, 1])

            X_axis_delay = np.arange(len(X))

            # Create the bar chart for payment delay churn rate
            fig_churn_delay = plt.figure(figsize=(10, 5))
            plt.bar(X_axis_delay - 0.2, churn_1_delay, 0.4, label='Churn')
            plt.bar(X_axis_delay + 0.2, churn_0_delay, 0.4, label='Not Churn')

            plt.xticks(X_axis_delay, X, rotation=90)
            plt.xlabel("Customer payment delays in days")
            plt.ylabel('Count')
            plt.title("Churn rate based on payment delays")
            plt.legend(loc='center right')
            plt.grid(axis='y')
            st.pyplot(fig_churn_delay)
            st.write("""
            **Note:** Customers who are not churned tend to have higher payment delay days as compared with churned customers till day 20
            """)
            
            filtered_support = df.groupby(['Support Calls', 'Churn']).size().unstack()

            X = list(filtered_support.index)
            churn_0_support = list(filtered_support.iloc[:, 0])
            churn_1_support = list(filtered_support.iloc[:, 1])

            X_axis_support = np.arange(len(X))

            # Create the bar chart for support calls churn rate
            fig_churn_support = plt.figure(figsize=(10, 5))
            plt.bar(X_axis_support - 0.2, churn_1_support, 0.4, label='Churn')
            plt.bar(X_axis_support + 0.2, churn_0_support, 0.4, label='Not Churn')

            plt.xticks(X_axis_support, X, rotation=45)
            plt.xlabel('Customer Support Calls')
            plt.ylabel('Count')
            plt.title("Churn rate based on support calls made by customers")
            plt.legend(loc='center right')
            plt.grid(axis='y')
            st.pyplot(fig_churn_support)
            st.write("""
            **Note:** customers with more support calls tend to churn more.

On the contrary, customers who are not churned tend to make much more 0 to 3 customer support calls than churned customers, after which churned customers make significantly more calls.
            """)
            
            
            
            
            
           

# 4. Upper navbar to navigate between pages
selected = option_menu(
    menu_title=None,
    options=["Home", "Insights", "Predict"],  
    icons=["house", "graph-up-arrow", "bar-chart-line"],  
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "nav-link": {
            "font-size": "20px",
            "text-align": "center",
            "margin": "0px",
            "width": "100%",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#0d6efd"}, 
    }
)

# 5. Render the selected page
if selected == "Home":
    home_page()
elif selected == "Predict":
    predict_page()
elif selected == "Insights":
    insights_page()
