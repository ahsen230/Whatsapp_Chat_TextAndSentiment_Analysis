import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import emoji
import random
import seaborn as sns
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import math

@st.cache_data
def run_sentiment_analysis(df, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
    """Perform sentiment analysis on the chat data and cache the results."""

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Perform sentiment analysis
    sentiments = []
    for message in df["Message"]:
        if isinstance(message, str):
            # Tokenize and pass through the model
            inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            scores = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
            sentiment = np.argmax(scores)
            sentiments.append(["Negative", "Neutral", "Positive"][sentiment])
        else:
            sentiments.append("Neutral")

    # Add sentiment column to DataFrame
    df["Sentiment"] = sentiments
    return df

 
def process_chat(chat_text):
    """
    Process WhatsApp chat text into a structured DataFrame.
    Handles multiple timestamp formats:
    - "MM/DD/YY, HH:MM PM - Sender: Message"
    - "[MM/DD/YY, HH:MM:SS PM] Sender: Message"
    
    Args:
        chat_text (str): The raw chat text to process
        
    Returns:
        pandas.DataFrame: DataFrame with columns ["Timestamp", "Sender", "Message"]
    """
    # Fixed pattern with proper escaping
    pattern = r'\[?(\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}(?::\d{2})?\s*[APap][Mm])\]?\s*-?\s*([^:]+):\s*(.+)'
    
    messages = re.findall(pattern, chat_text)
    
    # Create DataFrame if matches are found
    if messages:
        df = pd.DataFrame(messages, columns=["Timestamp", "Sender", "Message"])
        
        # Clean sender names (remove extra whitespace)
        df["Sender"] = df["Sender"].str.strip()
        
        # Clean timestamp strings: remove any square brackets
        df["Timestamp"] = df["Timestamp"].str.replace(r'[\[\]]', '', regex=True).str.strip()
        
        # Try multiple timestamp formats
        for fmt in [
            "%m/%d/%y, %I:%M:%S %p",  # Try the format with seconds first
            "%m/%d/%y, %I:%M %p",      # Then try without seconds
            "%m/%d/%Y, %I:%M:%S %p",   # 4-digit year with seconds
            "%m/%d/%Y, %I:%M %p"       # 4-digit year without seconds
        ]:
            try:
                # Convert Timestamp to datetime
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=fmt)
                break  # If successful, break the loop
            except ValueError:
                continue
                
        # Filter out media messages
        df = df[~df["Message"].str.contains("media omitted|image omitted", case=False, na=False)]
        df = df[~df["Sender"].str.contains("joined using this group's invite|~|messages and calls are end-to-end encrypted", case=False, na=False)]

        
        return df
    else:
        # Return an empty DataFrame if no matches
        return pd.DataFrame(columns=["Timestamp", "Sender", "Message"])


def calculate_metrics(df):
    """Calculate key metrics for the chat."""
    total_messages = len(df)  # Total messages
    unique_users = df["Sender"].nunique()  # Unique users
    messages_per_user = round(total_messages / unique_users, 2) if unique_users > 0 else 0
    most_active_user = df["Sender"].value_counts().idxmax() if not df.empty else "N/A"
    most_active_user_count = df["Sender"].value_counts().max() if not df.empty else 0
    streak_days = len(df["Timestamp"].dt.date.unique())  # Longest streak (days)
    return total_messages, unique_users, messages_per_user, most_active_user, most_active_user_count, streak_days



def total_messages_per_day(df, filtered_senders):
    """Plot total messages sent each day as a continuous line graph."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_per_day = df_filtered.groupby(df_filtered["Timestamp"].dt.date).size()
    plt.figure(figsize=(10, 8))
    messages_per_day.plot(kind='line', title='Total Messages Sent Each Day')
    
    # Reduce x-axis text size
    plt.xticks(fontsize=8, rotation=45)  # Smaller font size and rotated for better fit
    
    # Other formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout to prevent date labels from being cut off
    plt.tight_layout()
    
    st.pyplot(plt)


def total_messages_by_person(df, filtered_senders):
    """Plot total messages sent by each person."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_by_sender = df_filtered['Sender'].value_counts()

    plt.figure(figsize=(10, 8))
    ax = messages_by_sender.plot(kind='barh', color='skyblue', title='Total Messages Sent by Each Person')
    
    # Add data labels
    for i, v in enumerate(messages_by_sender):
        ax.text(v + 1, i, str(v), 
                va='center',           # Vertical alignment
                fontweight='bold',     # Make labels bold
                fontsize=10)           # Set font size
    
    plt.xlabel('Number of Messages')
    plt.ylabel('Sender')
    plt.gca().invert_yaxis()
    
    # Add some padding to the right to ensure labels are visible
    plt.margins(x=0.1)
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    
    st.pyplot(plt)

def total_messages_per_hour(df, filtered_senders):
    """Plot total messages sent per hour."""
def total_messages_per_hour(df, filtered_senders):
    """
    Plot total messages sent per hour with enhanced visualization and time period legend.
    Only shows hours that have messages.
    
    Args:
        df (pandas.DataFrame): DataFrame containing message data
        filtered_senders (list): List of senders to include in the analysis
    """
    def get_time_period_color(hour):
        """Return color and period name based on hour."""
        if 5 <= hour <= 11:
            return '#FFA07A', 'Morning (5:00-11:59)'    # Light salmon
        elif 12 <= hour <= 16:
            return '#98FB98', 'Afternoon (12:00-16:59)'  # Pale green
        elif 17 <= hour <= 21:
            return '#87CEEB', 'Evening (17:00-21:59)'    # Sky blue
        else:
            return '#DDA0DD', 'Night (22:00-4:59)'       # Plum
    
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_per_hour = df_filtered.groupby(df_filtered["Timestamp"].dt.hour).size()
    
    # Filter out hours with zero messages
    messages_per_hour = messages_per_hour[messages_per_hour > 0]
    
    plt.figure(figsize=(10, 8))
    
    # Create bars with different colors based on time period
    bars = plt.bar(range(len(messages_per_hour)), 
                  messages_per_hour,
                  alpha=0.7,
                  width=0.8)
    
    # Color each bar based on its time period and collect legend elements
    legend_elements = {}
    for i, (hour, count) in enumerate(messages_per_hour.items()):
        color, period = get_time_period_color(hour)
        bars[i].set_color(color)
        legend_elements[period] = color
        
        # Add data labels on top of each bar
        plt.text(i, count + 0.5, str(count),
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold')
    
    # Add legend
    legend_patches = [plt.Rectangle((0,0),1,1, fc=color, alpha=0.7) 
                     for color in legend_elements.values()]
    plt.legend(legend_patches, 
              legend_elements.keys(),
              loc='upper right',
              title='Time Periods',
              fontsize=8,
              title_fontsize=9)
    
    # Customize the plot
    plt.title('Message Activity by Hour of Day', pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Hour of Day (24-hour format)', fontsize=10)
    plt.ylabel('Number of Messages', fontsize=10)
    
    # Format x-axis ticks using only hours that have messages
    plt.xticks(range(len(messages_per_hour)), 
               [f'{hour:02d}:00' for hour in messages_per_hour.index],
               rotation=45)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add some padding
    plt.margins(x=0.01)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    st.pyplot(plt)

def messages_by_day_of_week(df, filtered_senders):
    """Plot messages sent by each day of the week."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_by_weekday = df_filtered.groupby(df_filtered["Timestamp"].dt.day_name()).size()
    
    # Standard day order
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Reindex and remove days with zero messages
    messages_by_weekday = messages_by_weekday.reindex(weekday_order).fillna(0)
    messages_by_weekday = messages_by_weekday[messages_by_weekday > 0]
    
    plt.figure(figsize=(10, 8))
    
    # Color mapping for weekdays/weekend
    def get_day_color(day):
        return '#87CEFA' if day in ['Saturday', 'Sunday'] else '#9370DB'
    
    # Create bars with different colors for weekdays/weekend
    bars = plt.bar(messages_by_weekday.index, 
                   messages_by_weekday.values,
                   color=[get_day_color(day) for day in messages_by_weekday.index],
                   alpha=0.7,
                   width=0.8)
    
    # Add data labels on top of each bar
    for i, v in enumerate(messages_by_weekday):
        plt.text(i, v + 0.5, str(int(v)),
                 ha='center', 
                 va='bottom',
                 fontsize=9,
                 fontweight='bold')
    
    # Customize the plot
    plt.title('Messages Sent by Day of Week', pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Day of Week', fontsize=10)
    plt.ylabel('Number of Messages', fontsize=10)
    
    # Add legend
    weekend_patch = plt.Rectangle((0,0),1,1, fc='#87CEFA', alpha=0.7, label='Weekend')
    weekday_patch = plt.Rectangle((0,0),1,1, fc='#9370DB', alpha=0.7, label='Weekday')
    plt.legend(handles=[weekday_patch, weekend_patch], 
               loc='upper right',
               title='Day Type',
               fontsize=8,
               title_fontsize=9)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(plt)


def top_words_distribution(df, filtered_senders):
    """
    Create dynamic grid of top words plots based on number of senders.
    
    Args:
        df (pandas.DataFrame): DataFrame containing message data
        filtered_senders (list): List of senders to analyze
    """
    # Determine number of columns dynamically
    if len(filtered_senders) > 6:
        ncols = 4
    elif len(filtered_senders) >= 5:
        ncols = 3
    elif len(filtered_senders) >= 3:
        ncols = 2
    else:
        ncols = 2
    
    # Calculate number of rows
    nrows = math.ceil(len(filtered_senders) / ncols)
    
    # Create figure with calculated subplot grid
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(4*ncols, 4*nrows), 
        squeeze=False  # Ensure axes is always 2D array
    )
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_senders)))
    
    # Process and plot for each sender
    for i, sender in enumerate(filtered_senders):
        # Filter and process sender's messages
        sender_df = df[df["Sender"] == sender]
        all_words = " ".join(sender_df["Message"].dropna().str.lower())
        
        words = re.findall(r'\b\w+\b', all_words)
        
        common_words = Counter(words).most_common(10)
        
        # Create subplot
        ax = axes_flat[i]
        
        # Sort words by count for horizontal bar plot
        words_series = pd.Series(dict(common_words)).sort_values(ascending=True)
        
        # Plot horizontal bars
        bars = ax.barh(words_series.index, words_series.values, color=colors[i], alpha=0.7)
        
        # Add labels and styling
        ax.set_title(f"{sender}", fontsize=8, fontweight='bold')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center', fontweight='bold', fontsize=6)
        
        # Style improvements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    # Overall figure styling
    plt.tight_layout()
    fig.suptitle('Top Words by Sender', fontsize=12, fontweight='bold')
    plt.subplots_adjust(top=0.9)  # Make room for overall title
    
    st.pyplot(fig)


def word_usage_visual(df):
    """Visualize word usage count against each person with a search filter."""
    # Create a text input for the search filter
    search_word = st.text_input("Search Word", placeholder="Enter a word to analyze its usage (default shows top word)...")

    # Preprocess: Extract all words from the chat
    all_messages = df[["Sender", "Message"]].dropna()
    all_words = []

    for sender, message in zip(all_messages["Sender"], all_messages["Message"]):
        words = re.findall(r'\b\w+\b', message.lower())  # Tokenize words (case insensitive)
        all_words.extend([(sender, word) for word in words])

    # Create a DataFrame with word counts for each sender
    words_df = pd.DataFrame(all_words, columns=["Sender", "Word"])
    word_counts = words_df.groupby(["Sender", "Word"]).size().reset_index(name="Count")

    # Determine the word to search
    if not search_word.strip():
        # Default: Show the most common word across all users
        top_word = word_counts.groupby("Word")["Count"].sum().idxmax()
        search_word = top_word

    # Filter the DataFrame for the search word
    filtered_df = word_counts[word_counts["Word"] == search_word.lower()]

    if filtered_df.empty:
        st.warning(f"No occurrences of the word '{search_word}' found.")
        return

    # Plot the results
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=filtered_df, x="Count", y="Sender", palette="Set3", orient="h")
    plt.title(f"Usage of the Word '{search_word}' by Each User")
    plt.xlabel("Count")
    plt.ylabel("Sender")

    # Add data labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center', xytext=(5, 0), textcoords='offset points',
                    fontweight='bold')

    st.pyplot(plt)

def plot_sentiment_distribution_per_user(df):
    """Plot Horizontal Bar Chart for Sentiment Distribution Per User."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Group and reshape data
    sentiment_user = df.groupby(["Sender", "Sentiment"]).size().unstack(fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.2  # Width of each bar
    y_positions = np.arange(len(sentiment_user.index))  # Y positions for users

    # Create bars for each sentiment
    ax.barh(y_positions - width, sentiment_user["Negative"], height=width, color="red", label="Negative")
    ax.barh(y_positions, sentiment_user["Neutral"], height=width, color="gray", label="Neutral")
    ax.barh(y_positions + width, sentiment_user["Positive"], height=width, color="green", label="Positive")

    # Add data labels
    for i, user in enumerate(sentiment_user.index):
        ax.text(sentiment_user["Negative"].iloc[i], y_positions[i] - width, str(sentiment_user["Negative"].iloc[i]),
                va='center', ha='left', fontsize=9)
        ax.text(sentiment_user["Neutral"].iloc[i], y_positions[i], str(sentiment_user["Neutral"].iloc[i]),
                va='center', ha='left', fontsize=9)
        ax.text(sentiment_user["Positive"].iloc[i], y_positions[i] + width, str(sentiment_user["Positive"].iloc[i]),
                va='center', ha='left', fontsize=9)

    # Customize chart appearance
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sentiment_user.index)
    ax.set_xlabel("Number of Messages")
    ax.set_ylabel("Senders")
    ax.set_title("Sentiment Distribution Per Sender")
    ax.legend(title="Sentiment")
    plt.tight_layout()

    # Display the chart in Streamlit
    st.pyplot(fig)

def display_top_sentiments(df, top_n=5):
    """Display the most positive and negative messages."""
    positive_messages = df[df["Sentiment"] == "Positive"].nlargest(top_n, "Sentiment_Num")
    negative_messages = df[df["Sentiment"] == "Negative"].nsmallest(top_n, "Sentiment_Num")

    st.subheader("Top Positive Messages")
    st.write(positive_messages[["Timestamp", "Sender", "Message"]])

    st.subheader("Top Negative Messages")
    st.write(negative_messages[["Timestamp", "Sender", "Message"]])

def plot_sentiment_pie_chart(df):
    """Generate a Plotly pie chart for the overall sentiment distribution."""
    
    # Count the sentiment distribution
    sentiment_counts = df["Sentiment"].value_counts()
    
    # Prepare the labels with both count and percentage
    labels = sentiment_counts.index
    counts = sentiment_counts.values
    percentages = (counts / counts.sum()) * 100
    label_text = [f"{label}: {count} ({percent:.1f}%)" for label, count, percent in zip(labels, counts, percentages)]
    
    # Create a Plotly pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,  # Sentiment types (Positive, Neutral, Negative)
        values=counts,  # Counts of each sentiment
        hole=0.3,  # Make it a donut chart (optional)
        hoverinfo="label+percent",  # Show both label and percentage on hover
        textinfo="text",  # Display the custom text (count and percentage)
        text=label_text,  # Custom text for each slice
        marker=dict(colors=["green", "gray", "red"])  # Colors for Positive, Neutral, Negative
    )])
    
    # Update layout for aesthetics
    fig.update_layout(
        title="Overall Sentiment Distribution",
        title_x=0.5,  # Center the title
        title_font=dict(size=20),
    )
    
    # Display the chart in Streamlit
    st.plotly_chart(fig)

url = "https://www.ahsenwaheed.com"
st.caption("Made by [me](%s) :)" % url)

def main():
    st.title("WhatsApp Chat Text Analytics 📊")
    st.divider()

    with st.expander("How to Download Whatsapp Chat"):
        st.image("https://mobi.easeus.com/images/en/screenshot/mobimover/export-whatsapp-chat-history.jpg")
        st.markdown("""
            - Open an individual or group chat.
            - Tap on the contact’s name or group name.
                - Select **Export Chat**.
                - If the chat contains media, choose:
                    - **Attach Media** (to include media files).
                    - **Without Media** (to exclude media files) **Use this method here**
        """)
        st.write("See WhatsApp documentation [here](https://faq.whatsapp.com/1180414079177245/) for more.") 

    uploaded_file = st.file_uploader("Upload chat file (txt)", type=["txt"])
    st.markdown("""
        <p style="font-size: 12px;">
        <strong>Disclaimer:</strong><br>
        The owner/creator of this application does <strong>not</strong> save any uploaded files. 
        Files are uploaded exclusively through the <code>st.file_uploader</code> function and are processed solely within the app. 
        Once the session ends, the files are discarded. 
        <strong>No data is stored or retained beyond the session</strong>.
        </p>
    """, unsafe_allow_html=True)



    
    if uploaded_file:
        # Read the uploaded file
        chat_text = uploaded_file.read().decode("utf-8")
        df = process_chat(chat_text)

        # Sidebar for sender selection
        st.sidebar.header("Filter by Sender")
        all_senders = df["Sender"].unique()
        selected_senders = st.sidebar.multiselect("Select Senders:", options=all_senders, default=all_senders)



        # Add tabs to organize the content
        tab1, tab2, tab3 = st.tabs(["Chat Preview", "Text Analysis", "Sentiment Analysis"])

        # Tab 1: Chat Preview
        with tab1:
            st.subheader("Chat Preview")
            if uploaded_file:
                # Show the first 300 characters of the uploaded chat
                st.text(chat_text[:300])  # Display top 300 characters from the chat

                # Show the processed DataFrame
                st.write("Processed Chat DataFrame:")
                st.write(df)
            else:
                st.warning("Please upload a chat file to preview the content.")

        # Tab 2: Text Analysis
        with tab2:
            st.subheader("Text Analysis")
            # Show metrics
            total_messages, unique_users, messages_per_user, most_active_user, most_active_user_count, streak_days = calculate_metrics(df)

            # Metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", total_messages ,border=True)
                st.metric("Unique Users", unique_users ,border=True)
            with col2:
                st.metric("Messages Per User", messages_per_user ,border=True)
                st.metric("Longest Streak (Days)", streak_days ,border=True)
            with col3:
                st.metric("Most Active User", most_active_user ,border=True)
                #st.metric(f"Messages by {most_active_user}", most_active_user_count)

            # Dashboard Layout

            # Row 1: Total Messages per Day and Total Messages by Person
            col4, col5 = st.columns(2)
            with col4:
                st.subheader("Total Messages Sent Each Day")
                total_messages_per_day(df, selected_senders)
            with col5:
                st.subheader("Total Messages Sent by Each Person")
                total_messages_by_person(df, selected_senders)

            # Row 2: Total Messages per Hour and Messages by Day of the Week
            col6, col7 = st.columns(2)
            with col6:
                st.subheader("Total Messages Per Hour")
                total_messages_per_hour(df, selected_senders)
            with col7:
                st.subheader("Messages Sent by Days")
                messages_by_day_of_week(df, selected_senders)

            # Row 3: Top Words Distribution
            st.subheader("Top 5 Most Used Words Distribution per Person")
            top_words_distribution(df, selected_senders)

            # Word Usage Visual (with independent search)
            st.subheader("Word Usage with Search Filter")
            word_usage_visual(df)

        # Tab 3: Sentiment Analysis
        with tab3:
            st.subheader("Sentiment Analysis")

            if df.empty:
                st.warning("Please upload a chat file to perform sentiment analysis.")
            else:
                df_with_sentiment = run_sentiment_analysis(df)


            plot_sentiment_pie_chart(df_with_sentiment)

            st.write(df_with_sentiment)

            sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
            df_with_sentiment["Sentiment_Num"] = df_with_sentiment["Sentiment"].map(sentiment_map)
            df_with_sentiment["Sentiment_Num"] = pd.to_numeric(df_with_sentiment["Sentiment_Num"], errors="coerce").fillna(0)

            st.subheader("Sentiment Distribution per user")
            plot_sentiment_distribution_per_user(df_with_sentiment)
            st.divider()
            # Most positive and most negative users
            positive_counts = df_with_sentiment[df_with_sentiment["Sentiment"] == "Positive"]["Sender"].value_counts()
            negative_counts = df_with_sentiment[df_with_sentiment["Sentiment"] == "Negative"]["Sender"].value_counts()
            neutral_counts = df_with_sentiment[df_with_sentiment["Sentiment"] == "Neutral"]["Sender"].value_counts()
            most_positive_user = positive_counts.idxmax() if not positive_counts.empty else "No Data"
            most_negative_user = negative_counts.idxmax() if not negative_counts.empty else "No Data"
            most_neutral_user = neutral_counts.idxmax() if not neutral_counts.empty else "No Data"

            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Most Positive User 😊", most_positive_user, border=True)
            with col4:
                st.metric("Most Negative User 😤", most_negative_user, border=True)
            with col5:
                st.metric("Most Neutral User 😐", most_neutral_user, border=True)


            st.subheader("Key Moments in Chat" , divider="gray")
            display_top_sentiments(df_with_sentiment, top_n=5)


if __name__ == "__main__":
    main()
