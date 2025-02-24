import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import email

# st.markdown(f"""
# <style>
# {open('style.css', 'r').read()}
# </style>
# """, unsafe_allow_html=True)
# Load data
data = pd.read_csv('machinery_data.csv')
data.bfill()

# Feature selection and normalization
features = ['vibration', 'temperature', 'rotating_speed', 'current', 'operational_hours']
target_rul = 'RUL'
target_maintenance = 'maintenance'
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Split data for regression and classification
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data[features], data[target_rul], test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(data[features], data[target_maintenance], test_size=0.2, random_state=42)

# Train models
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(data[features])

# Prediction function

def predict_maintenance(features):
    rul_pred = reg_model.predict([features])
    maint_pred = clf_model.predict([features])
    cluster_pred = kmeans.predict([features])
    return {
        'RUL Prediction': rul_pred[0],
        'Maintenance Prediction': 'Needs Maintenance' if maint_pred[0] == 1 else 'No Maintenance Is Needed',
        'Anomaly Detection': 'Anomaly' if cluster_pred[0] == 1 else 'Normal'
    }
# Define a function to send an email notification
def send_email_notification(receiver_email, subject, body):
    sender_email = "adduu48@gmail.com"  # Replace with your email address
    sender_password = "vaqoilkdyexeyuvr"  # Replace with your email password or app password
    
    # Set up the MIME message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    
    try:
        # Set up the SMTP server and send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            st.success(f"Email notification sent to {receiver_email}")
    except Exception as e:
        st.error(f"Failed to send email. Error: {str(e)}")
    
# Streamlit Option Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Input Data", "Results", "Historical Data", "Visualizations", "Maintenance Log", "Maintenance History", "Maintenance Schedule", "Email Notifications"],
        icons=["house", "input-cursor", "check2-circle",  "table", "bar-chart-line", "book", "robot", "clock", "envelope"],
        menu_icon="cast",
        default_index=0,
    )

# Home section
if selected == "Home":
    # Page title and introduction
    st.title("Welcome to the Predictive Maintenance Dashboard")
    st.markdown(
        """
        **This application provides predictive maintenance insights for industrial machinery.**
        
        Leverage the power of predictive analytics to reduce downtime, improve machine reliability, and optimize maintenance strategies.
        """
    )

    # Horizontal divider for neat sectioning
    st.divider()

    # Predictive Analysis section
    st.header("🔍 Predictive Analysis")
    st.markdown(
        """
        **Plan ahead with predictive analytics.** 
        Our system helps you schedule timely maintenance, reducing the likelihood of unexpected breakdowns. 
        With data-driven insights, optimize maintenance processes and extend the lifespan of your equipment.
        """
    )

    # Example of a static placeholder image or a plot (could be a graph in a real use case)
    st.image("Predictive-Maintenance-768x375.jpg")

    # Divider for the next section
    st.divider()

    # Cost Reduction section
    st.header("💰 Cost Reduction")
    st.markdown(
        """
        **Save costs with smarter maintenance.** 
        By using predictive maintenance, you can:
        
        - Prevent unnecessary repairs.
        - Reduce equipment downtime.
        - Increase operational efficiency, saving thousands in the long run.
        """
    )

    # Add a static image related to cost savings or financial benefits
    st.image("COST-Saving-KERJAINDO-1024x682.jpeg")

    # Final horizontal divider for closing the section
    st.divider()

    # Concluding message
    st.markdown(
        """
        Explore the app using the navigation menu to gain deeper insights into equipment health, performance trends, 
        and customized maintenance recommendations.
        """
    )


elif selected == "Input Data":
    st.title("🔧 Input Features")
    st.markdown("Use the sliders to input the sensor readings and operational hours or generate random values.")

    if 'generated_values' not in st.session_state:
        st.session_state['generated_values'] = None

    if st.button('Generate Random Values'):
        vibration = np.random.uniform(data['vibration'].min(), data['vibration'].max())
        temperature = np.random.uniform(data['temperature'].min(), data['temperature'].max())
        rotating_speed = np.random.uniform(data['rotating_speed'].min(), data['rotating_speed'].max())
        current = np.random.uniform(data['current'].min(), data['current'].max())
        operational_hours = np.random.uniform(data['operational_hours'].min(), data['operational_hours'].max())
        st.session_state['generated_values'] = [vibration, temperature, rotating_speed, current, operational_hours]
        st.success("Random values generated successfully!")

    if st.session_state['generated_values'] is not None:
        st.write("**Generated Values:**")
        st.write(f"Vibration: {st.session_state['generated_values'][0]:.2f}")
        st.write(f"Temperature: {st.session_state['generated_values'][1]:.2f}")
        st.write(f"Rotating Speed: {st.session_state['generated_values'][2]:.2f}")
        st.write(f"Current: {st.session_state['generated_values'][3]:.2f}")
        st.write(f"Operational Hours: {st.session_state['generated_values'][4]:.2f}")

        if st.button('Use Generated Values'):
            st.session_state['input_features'] = st.session_state['generated_values']
            st.success("Generated values have been used. Navigate to the Results page to see the predictions.")

    st.markdown("**Or manually input values:**")
    vibration = st.slider('Vibration', float(data['vibration'].min()), float(data['vibration'].max()), float(data['vibration'].mean()))
    temperature = st.slider('Temperature', float(data['temperature'].min()), float(data['temperature'].max()), float(data['temperature'].mean()))
    rotating_speed = st.slider('Rotating Speed', float(data['rotating_speed'].min()), float(data['rotating_speed'].max()), float(data['rotating_speed'].mean()))
    current = st.slider('Current', float(data['current'].min()), float(data['current'].max()), float(data['current'].mean()))
    operational_hours = st.slider('Operational Hours', int(data['operational_hours'].min()), int(data['operational_hours'].max()), int(data['operational_hours'].mean()))

    if st.button('Submit'):
        st.session_state['input_features'] = [vibration, temperature, rotating_speed, current, operational_hours]
        st.success("Input data submitted successfully! Navigate to the Results page to see the predictions.")


elif selected == "Results":
    st.title("📊 Prediction Results")
    if 'input_features' not in st.session_state:
        st.warning("Please input data first in the 'Input Data' section.")
    else:
        input_features = pd.Series(st.session_state['input_features'], index=features)  # Ensure this is a Series
        prediction = predict_maintenance(input_features)
        
        st.subheader(f"**Remaining Useful Life (RUL):** {prediction['RUL Prediction']:.2f} hours")
        st.subheader(f"**Maintenance Status:** {prediction['Maintenance Prediction']}")
        st.subheader(f"**Anomaly Detection:** {prediction['Anomaly Detection']}")
        
        if prediction['Maintenance Prediction'] == 'Needs Maintenance':
            st.error('⚠️ Maintenance is required!')
        
        if prediction['Anomaly Detection'] == 'Anomaly':
            st.warning('⚠️ Anomaly detected in sensor readings!')
            
            # Send email notification if an anomaly is detected
            if 'receiver_email' in st.session_state:
                subject = "Anomaly Detected in Predictive Maintenance System"
                body = f"""
                Anomaly detected in the predictive maintenance system. 
                Details:
                - Remaining Useful Life (RUL): {prediction['RUL Prediction']:.2f} hours
                - Maintenance Status: {prediction['Maintenance Prediction']}
                - Anomaly Detection: {prediction['Anomaly Detection']}
                
                Please take appropriate actions immediately.
                """
                send_email_notification(st.session_state['receiver_email'], subject, body)
            else:
                st.warning("No email address configured for notifications.")

elif selected == "Historical Data":
    st.title("📂 Historical Data")
    st.write(data.head(100))
    
elif selected == "Maintenance Log":
    st.title("📅 Maintenance Log")

    # Input fields
    maintenance_date = st.date_input('Maintenance Date')
    maintenance_hours = st.number_input('Operational Hours at Maintenance')
    maintenance_type = st.selectbox('Maintenance Type', [' ', 'Minor', 'Major', 'Preventive', 'Reactive', 'Predictive'], index=0)
    sensor_type = st.selectbox('Sensor Type', [' ', 'Temperature', 'Vibration', 'Speed', 'Current'], index=0)
    comments = st.text_area('Comments')

    # Check if user has selected a valid maintenance type and sensor type
    if st.button('Submit Maintenance Log'):
        if maintenance_type == ' ' or sensor_type == ' ' or maintenance_type == ' ':
            st.error("Please select a valid maintenance type and sensor type.")
        else:
            # Add data to a log (could be a CSV or database)
            new_log = pd.DataFrame({
                'Date': [maintenance_date],
                'Operational Hours': [maintenance_hours],
                'Maintenance Type': [maintenance_type],
                'Sensor Type': [sensor_type],
                'Comments': [comments]
            })
            
            # Check if file exists to include headers only the first time
            file_exists = os.path.isfile('maintenance_log.csv')
            
            # Append the log to CSV
            new_log.to_csv('maintenance_log.csv', mode='a', header=not file_exists, index=False)
            
            # Show success message
            st.success('Maintenance log added successfully!')
            
            # Display the newly added log
            st.write("Newly added maintenance log:")
            st.write(new_log)

elif selected == "Maintenance History":
    st.title("🗒 Maintenance History")
    
    new_log = pd.DataFrame({
                'Date': [],
                'Operational Hours': [],
                'Maintenance Type': [],
                'Sensor Type': [],
                'Comments': []
            })
    
        # Try to read the CSV file
    maintenance_log = pd.read_csv('maintenance_log.csv')
    st.write(maintenance_log)

elif selected == "Maintenance Schedule":
    st.title("🛠️ Maintenance Scheduling")
    
    # Check if input data exists
    if 'input_features' not in st.session_state:
        st.warning("Please input data first in the 'Input Data' section.")
    else:
        # Convert session state input features to a pandas Series
        input_features = pd.Series(st.session_state['input_features'], index=features)
        
        # Attempt to get the RUL prediction using the model
        try:
            # Predict Remaining Useful Life (RUL)
            rul_prediction = predict_maintenance(input_features)['RUL Prediction']
            
            # Allow users to specify a custom RUL threshold for scheduling maintenance
            maintenance_threshold = 150
            
            # Calculate days and hours from RUL prediction
            days = rul_prediction // 24  # Get complete days from RUL
            hours = rul_prediction % 24   # Get remaining hours from RUL
            
            # Calculate the next maintenance date based on RUL prediction
            next_maintenance_date = datetime.now() + timedelta(hours=rul_prediction)
            
            # Display results
            st.subheader(f"**Predicted RUL:** {int(days)} days and {int(hours)} hours")
            st.subheader(f"**Suggested Maintenance Date:** {next_maintenance_date.strftime('%A, %B %d, %Y')}")
            
            # Machine health indicator based on RUL
            health_status = "Critical" if rul_prediction < maintenance_threshold else "Good"
            st.metric(label="Machine Health Status:", value=health_status)
            
            # Notify user if maintenance is recommended soon
            if rul_prediction < maintenance_threshold:
                st.error(f"⚠️ Maintenance is required soon! RUL is below the threshold of {maintenance_threshold} hours.")
            else:
                st.success(f"✅ Machine is operating normally. No immediate maintenance required.")
        
        except KeyError as e:
            # Handle specific errors related to prediction
            st.error(f"Error in prediction: {e}")
        except Exception as e:
            # Handle any other unexpected errors
            st.error(f"An unexpected error occurred: {e}")
    
elif selected == "Email Notifications":
    st.title("📧 Email Notifications")
    
    # User input for receiving email address
    receiver_email = st.text_input("Enter the email address to receive notifications:")
    if st.button("Save Email"):
        st.session_state['receiver_email'] = receiver_email
        st.success("Email address saved successfully!")
    
    # Display current email address (if any)
    if 'receiver_email' in st.session_state:
        st.write(f"**Current Notification Email:** {st.session_state['receiver_email']}")
    else:
        st.warning("No email address set yet.")

elif selected == "Visualizations":
    st.title("📊 Data Visualizations")

    # Histogram for sensor readings
    st.subheader("Histogram of Sensor Readings")
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    sns.histplot(data['vibration'], bins=30, ax=axs[0], kde=True)
    axs[0].set_title('Vibration')
    sns.histplot(data['temperature'], bins=30, ax=axs[1], kde=True)
    axs[1].set_title('Temperature')
    sns.histplot(data['rotating_speed'], bins=30, ax=axs[2], kde=True)
    axs[2].set_title('Rotating Speed')
    sns.histplot(data['current'], bins=30, ax=axs[3], kde=True)
    axs[3].set_title('Current')
    st.pyplot(fig)

    # Scatter plot for sensor readings vs operational hours
    st.subheader("Scatter Plot of Sensor Readings vs Operational Hours")
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].scatter(data['operational_hours'], data['vibration'], alpha=0.5)
    axs[0].set_title('Operational Hours vs Vibration')
    axs[0].set_xlabel('Operational Hours')
    axs[0].set_ylabel('Vibration')
    axs[1].scatter(data['operational_hours'], data['temperature'], alpha=0.5)
    axs[1].set_title('Operational Hours vs Temperature')
    axs[1].set_xlabel('Operational Hours')
    axs[1].set_ylabel('Temperature')
    axs[2].scatter(data['operational_hours'], data['rotating_speed'], alpha=0.5)
    axs[2].set_title('Operational Hours vs Rotating Speed')
    axs[2].set_xlabel('Operational Hours')
    axs[2].set_ylabel('Rotating Speed')
    axs[3].scatter(data['operational_hours'], data['current'], alpha=0.5)
    axs[3].set_title('Operational Hours vs Current')
    axs[3].set_xlabel('Operational Hours')
    axs[3].set_ylabel('Current')
    st.pyplot(fig)

    # Line chart for RUL over time
    st.subheader("Line Chart of RUL Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-')
    ax.set_title('RUL Over Operational Hours')
    ax.set_xlabel('Operational Hours')
    ax.set_ylabel('RUL')
    st.pyplot(fig)

    if 'input_features' in st.session_state:
        input_features = st.session_state['input_features']

        # Overlay generated input values if available
        if input_features is not None:
            # Histogram for sensor readings with generated input
            st.subheader("Histogram of Sensor Readings with Generated Input")
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            sns.histplot(data['vibration'], bins=30, ax=axs[0], kde=True)
            axs[0].set_title('Vibration')
            axs[0].axvline(input_features[0], color='red', linestyle='--', label='Generated Value')
            sns.histplot(data['temperature'], bins=30, ax=axs[1], kde=True)
            axs[1].set_title('Temperature')
            axs[1].axvline(input_features[1], color='red', linestyle='--', label='Generated Value')
            sns.histplot(data['rotating_speed'], bins=30, ax=axs[2], kde=True)
            axs[2].set_title('Rotating Speed')
            axs[2].axvline(input_features[2], color='red', linestyle='--', label='Generated Value')
            sns.histplot(data['current'], bins=30, ax=axs[3], kde=True)
            axs[3].set_title('Current')
            axs[3].axvline(input_features[3], color='red', linestyle='--', label='Generated Value')
            plt.legend()
            st.pyplot(fig)

            # Scatter plot for sensor readings vs operational hours with generated input
            st.subheader("Scatter Plot of Sensor Readings vs Operational Hours with Generated Input")
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            axs[0].scatter(data['operational_hours'], data['vibration'], alpha=0.5)
            axs[0].set_title('Operational Hours vs Vibration')
            axs[0].set_xlabel('Operational Hours')
            axs[0].set_ylabel('Vibration')
            axs[0].axvline(input_features[4], color='red', linestyle='--', label='Generated Value')
            axs[0].legend()
            axs[1].scatter(data['operational_hours'], data['temperature'], alpha=0.5)
            axs[1].set_title('Operational Hours vs Temperature')
            axs[1].set_xlabel('Operational Hours')
            axs[1].set_ylabel('Temperature')
            axs[1].axvline(input_features[4], color='red', linestyle='--', label='Generated Value')
            axs[1].legend()
            axs[2].scatter(data['operational_hours'], data['rotating_speed'], alpha=0.5)
            axs[2].set_title('Operational Hours vs Rotating Speed')
            axs[2].set_xlabel('Operational Hours')
            axs[2].set_ylabel('Rotating Speed')
            axs[2].axvline(input_features[4], color='red', linestyle='--', label='Generated Value')
            axs[2].legend()
            axs[3].scatter(data['operational_hours'], data['current'], alpha=0.5)
            axs[3].set_title('Operational Hours vs Current')
            axs[3].set_xlabel('Operational Hours')
            axs[3].set_ylabel('Current')
            axs[3].axvline(input_features[4], color='red', linestyle='--', label='Generated Value')
            axs[3].legend()
            st.pyplot(fig)

            # Line chart for RUL over time with generated input
            st.subheader("Line Chart of RUL Over Time with Generated Input")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data['operational_hours'], data['RUL'], marker='o', linestyle='-')
            ax.set_title('RUL Over Operational Hours')
            ax.set_xlabel('Operational Hours')
            ax.set_ylabel('RUL')
            ax.axvline(input_features[4], color='red', linestyle='--', label='Generated Value')
            ax.legend()
            st.pyplot(fig)
