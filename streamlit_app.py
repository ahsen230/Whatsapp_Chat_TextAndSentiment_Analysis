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
    """Process WhatsApp chat text into a structured DataFrame."""
    # Updated Regex for WhatsApp messages
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\u202f[APap][Mm]) - ([^:]+): (.+)"
    messages = re.findall(pattern, chat_text)

    # Create DataFrame if matches are found
    if messages:
        df = pd.DataFrame(messages, columns=["Timestamp", "Sender", "Message"])
        # Convert Timestamp to datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%y, %I:%M %p", errors="coerce")
        df = df[df["Message"] != "<Media omitted>"]
    else:
        # Return an empty DataFrame if no matches
        df = pd.DataFrame(columns=["Timestamp", "Sender", "Message"])
    
    return df

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

    plt.figure(figsize=(10, 6))
    messages_per_day.plot(kind='line', title='Total Messages Sent Each Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.grid(True, linestyle='--', alpha=0.6)  # Optional: Add light grid for readability
    st.pyplot(plt)


def total_messages_by_person(df, filtered_senders):
    """Plot total messages sent by each person."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_by_sender = df_filtered['Sender'].value_counts()

    plt.figure(figsize=(10, 7))
    messages_by_sender.plot(kind='barh', color='skyblue', title='Total Messages Sent by Each Person')
    plt.xlabel('Number of Messages')
    plt.ylabel('Sender')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

def total_messages_per_hour(df, filtered_senders):
    """Plot total messages sent per hour."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_per_hour = df_filtered.groupby(df_filtered["Timestamp"].dt.hour).size()

    plt.figure(figsize=(10, 7))
    messages_per_hour.plot(kind='bar', color='orange', title='Total Messages Sent by Each Hour')
    plt.xlabel('Hour')
    plt.ylabel('Number of Messages')
    plt.xticks(range(0, 24))
    st.pyplot(plt)

def messages_by_day_of_week(df, filtered_senders):
    """Plot messages sent by each day of the week."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_by_weekday = df_filtered.groupby(df_filtered["Timestamp"].dt.day_name()).size()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    messages_by_weekday = messages_by_weekday.reindex(weekday_order)

    plt.figure(figsize=(10, 6))
    messages_by_weekday.plot(kind='bar', color='purple', title='Messages Sent by Each Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Messages')
    st.pyplot(plt)

def most_used_words(df, filtered_senders):
    """Show most used words per person as a horizontal bar plot."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    all_messages = " ".join(df_filtered["Message"].dropna().str.lower())
    words = re.findall(r'\b\w+\b', all_messages)  # Extract words
    common_words = Counter(words).most_common(10)

    # Convert to DataFrame for plotting
    words_df = pd.DataFrame(common_words, columns=["Word", "Count"])

    # Plot the horizontal bar chart
    plt.figure(figsize=(10, 5))
    words_df.set_index("Word")["Count"].sort_values(ascending=False).plot(
        kind='barh', color='orange', title='Top 10 Most Used Words'
    )
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most frequent word at the top
    st.pyplot(plt)


def top_words_distribution(df, filtered_senders):
    """Facet-style horizontal bar plots for the top 10 most used words per sender with data labels."""
    # Filter data for the selected senders
    df_filtered = df[df['Sender'].isin(filtered_senders)]

    # Create a DataFrame for storing all senders' word counts
    data = []

    # Extract the top 10 most used words for each sender
    for sender in filtered_senders:
        sender_df = df_filtered[df_filtered["Sender"] == sender]
        all_words = " ".join(sender_df["Message"].dropna().str.lower())
        words = re.findall(r'\b\w+\b', all_words)  # Extract words
        common_words = Counter(words).most_common(10)  # Top 10 words
        for word, count in common_words:
            data.append({"Sender": sender, "Word": word, "Count": count})

    # Create a new DataFrame for visualization
    words_df = pd.DataFrame(data)

    if words_df.empty:
        st.write("No data available for the selected users.")
        return

    # Set the style and plot with seaborn
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=words_df,
        x="Count",
        y="Word",
        hue="Sender",
        col="Sender",
        kind="bar",
        col_wrap=4,  # Adjust this for layout (e.g., number of columns per row)
        height=4,
        sharey=False,  # Allow different y-scales for each sender
        palette="Set2"  # Color palette for unique sender colors
    )

    # Add data labels to each bar
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", label_type="edge", fontsize=10, padding=3)

    # Customize plot appearance
    g.set_titles("{col_name}")  # Set titles for each sender
    g.set_axis_labels("Frequency", "Words")  # Axis labels
    g.fig.subplots_adjust(top=0.9)  # Adjust subplot spacing
    g.fig.suptitle("Most Often Used Words by Each Sender")  # Overall title

    st.pyplot(g.fig)


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
    sns.barplot(data=filtered_df, x="Count", y="Sender", palette="Set3", orient="h")
    plt.title(f"Usage of the Word '{search_word}' by Each User")
    plt.xlabel("Count")
    plt.ylabel("User")
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
    ax.set_ylabel("Users")
    ax.set_title("Sentiment Distribution Per User")
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
