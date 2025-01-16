import streamlit as st
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import emoji

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

def total_messages_per_day(df, filtered_senders):
    """Plot total messages sent each day."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_per_day = df_filtered.groupby(df_filtered["Timestamp"].dt.date).size()

    plt.figure(figsize=(10, 5))
    messages_per_day.plot(kind='line', marker='o', title='Total Messages Sent Each Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.grid()
    st.pyplot(plt)

def total_messages_by_person(df):
    """Plot total messages sent by each person."""
    messages_by_sender = df['Sender'].value_counts()

    plt.figure(figsize=(10, 5))
    messages_by_sender.plot(kind='barh', color='skyblue', title='Total Messages Sent by Each Person')
    plt.xlabel('Number of Messages')
    plt.ylabel('Sender')
    st.pyplot(plt)

def total_messages_per_hour(df, filtered_senders):
    """Plot total messages sent per hour."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    messages_per_hour = df_filtered.groupby(df_filtered["Timestamp"].dt.hour).size()

    plt.figure(figsize=(10, 5))
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

    plt.figure(figsize=(10, 5))
    messages_by_weekday.plot(kind='bar', color='purple', title='Messages Sent by Each Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Messages')
    st.pyplot(plt)

def most_used_words(df, filtered_senders):
    """Show most used words per person."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    all_messages = " ".join(df_filtered["Message"].dropna().str.lower())
    words = re.findall(r'\b\w+\b', all_messages)  # Extract words
    common_words = Counter(words).most_common(10)

    st.write("**Most Used Words:**")
    for word, count in common_words:
        st.write(f"{word}: {count}")

def emoji_count_distribution(df, filtered_senders):
    """Show top 5 emoji count distribution against each person."""
    df_filtered = df[df['Sender'].isin(filtered_senders)]
    emoji_counter = Counter()

    for message in df_filtered["Message"]:
        emoji_counter.update(c for c in message if c in emoji.UNICODE_EMOJI["en"])

    most_common_emojis = emoji_counter.most_common(5)
    emoji_df = pd.DataFrame(most_common_emojis, columns=["Emoji", "Count"])

    st.write("**Top 5 Emojis and Their Counts:**")
    st.bar_chart(emoji_df.set_index("Emoji"))

def main():
    st.title("WhatsApp Chat Text Analytics")
    st.divider()

    st.subheader("Steps to download WhatsApp chat")
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
    st.divider()

    uploaded_file = st.file_uploader("Upload chat file (txt)", type=["txt"])

    if uploaded_file:
        # Read the uploaded file
        chat_text = uploaded_file.read().decode("utf-8")
        df = process_chat(chat_text)

        st.sidebar.header("Filter by Sender")
        all_senders = df["Sender"].unique()
        selected_senders = st.sidebar.multiselect("Select Senders:", options=all_senders, default=all_senders)

        # Visualizations
        st.subheader("Total Messages Sent Each Day")
        total_messages_per_day(df, selected_senders)

        st.subheader("Total Messages Sent by Each Person")
        total_messages_by_person(df)

        st.subheader("Total Messages Per Hour")
        total_messages_per_hour(df, selected_senders)

        st.subheader("Messages Sent by Each Day of the Week")
        messages_by_day_of_week(df, selected_senders)

        st.subheader("Most Used Words")
        most_used_words(df, selected_senders)

        st.subheader("Emoji Count Distribution")
        emoji_count_distribution(df, selected_senders)

if __name__ == "__main__":
    main()
