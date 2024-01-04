import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Initialize the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Sample DataFrame with questions and answers
# data = {'Question': ['What is AI?', 'How to learn Python?', 'What is machine learning?'], 
#         'Answer': ['AI is the field of study of intelligent agents.', 'You can learn Python by practicing regularly.', 'Machine learning is a subset of AI that focuses on data.']}
# df = pd.DataFrame(data)

df = pd.read_csv("qna.csv")

# Encode the questions from the DataFrame
question_embeddings = model.encode(df['Question'].tolist(), show_progress_bar=False, normalize_embeddings=True)

# Fit KNN to the question embeddings
knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
knn.fit(question_embeddings)

# Function to find answer to a new question
def get_answer(new_question):
    new_question_embedding = model.encode([new_question], normalize_embeddings=True)
    _, index = knn.kneighbors(new_question_embedding)
    return df.iloc[index[0][0]]['Answer']

# Streamlit App
st.title("Ask a Dentist")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    
    if message["role"] == 'user':
        with st.chat_message('user', avatar='😐'):
            st.markdown(message["content"])
    else:
        with st.chat_message('assistant', avatar='👩‍⚕️'):
            st.markdown(message["content"])
# React to user input
if prompt := st.chat_input("Ask your question:"):
    # Display user message in chat message container
    st.chat_message("user", avatar='😐').markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the answer to the user's question
    answer = get_answer(prompt)
    response = f"{answer}, does that help with your questions?"
    
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar= '👩‍⚕️'):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

url = 'https://portal.ada.org.au/Portal/Shared_Content/Smart-Suite/Smart-Maps/Public/Find-a-Dentist.aspx'
st.markdown(f"""
    <style>
        .find-dentist-button {{
            background-color: lightgrey;
            color: black;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            text-align: right;
        }}
    </style>
    <div style="text-align: right;">
        <a href="{url}" target="_blank">
            <button class="find-dentist-button">Find a Dentist</button>
        </a>
    </div>
    """, unsafe_allow_html=True)