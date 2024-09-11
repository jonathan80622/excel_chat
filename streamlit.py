import streamlit as st
from openai import OpenAI
import pickle
import numpy as np
import numpy.linalg as npla

def normalize(arr):
    return arr/npla.norm(arr)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

with open('oai_embed_excel.pkl', 'rb') as f:
    DF = pickle.load(f)

DF['embedding'] = DF['embedding'].apply(normalize)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_embedding(text):
    embedding_response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(embedding_response.data[0].embedding)
def calculate_similarity(prompt, df, top_n=5):
    query_embedding = get_embedding(prompt)
    df['similarity'] = df['embedding'].apply(lambda x: np.dot(query_embedding, x))
    return df.sort_values(by='similarity', ascending=False).head(top_n)
def query_df_with_rag(prompt, DF):
    # Calculate the top N similar results
    top_similar_results = calculate_similarity(prompt, DF, top_n=5)
    
    # Combine the top results into a single string or use the top result only
    retrieved_info = "\n\n".join(top_similar_results['content'].tolist())
    
    return retrieved_info if retrieved_info else None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the DataFrame for relevant information
    retrieved_info = calculate_similarity(prompt, DF, top_n=5)
    
    # Display retrieved information in the sidebar for verification
    st.sidebar.title("Retrieved Content for Verification")
    
    # Iterate over the rows and display the 'batch' content and similarity score
    for idx, (tpe, row) in enumerate(retrieved_info.iterrows()):
        batch_text = row['batch']  # Access 'batch' column since it contains the data
        st.sidebar.markdown(f"**Batch {idx+1}:** {batch_text}")
        st.sidebar.markdown(f"**Similarity:** {row['similarity']}")
        st.sidebar.markdown("---")

    # Combine the retrieved info into a string to be included in the assistant response
    combined_info = "\n\n".join(retrieved_info['batch'].tolist())

    assistant_response = ""
    if combined_info:
        assistant_response += f"**Retrieved Info:** {combined_info}\n\n"
    
    # Get assistant response from the model and include the retrieved info
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        generated_response = st.write_stream(stream)
        assistant_response += generated_response  # Combine with OpenAI model output

        # Display combined assistant response
        st.markdown(assistant_response)

    # Add the assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})