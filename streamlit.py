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

DF['Embedding'] = DF['Embedding'].apply(normalize)

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
    df['similarity'] = df['Embedding'].apply(lambda x: np.dot(query_embedding, x))
    return df.sort_values(by='similarity', ascending=False).head(top_n)

def get_related_cells_subset(cell_ref, df):
    queried_row = df[df['Cell Reference'] == cell_ref]

    if queried_row.empty:
        return f"Cell {cell_ref} not found."

    # Retrieve related cells from the queried cell
    related_cells = queried_row['Related cells'].values[0]

    # Create a subset of the DataFrame that includes the queried cell and related cells
    subset = df[df['Cell Reference'].apply(lambda r:np.any([(_ in r) for _ in [cell_ref] + related_cells]))]
    return subset

def concat_df(df):
    concatenated_descriptions = ""
    for _, row in df.iterrows():
        col_name = row['Cell Reference'][:1]  # Extract column letter from cell reference (e.g., 'A' from 'A1')
        cell_value = row['Value']
        description = row['Description']
        data = row['Data']
        concatenated_descriptions += f"[Column {col_name}, Cell {row['Cell Reference']} - Value: {cell_value}, Description: {description}, Data: {data}]\n"

    return concatenated_descriptions.strip()

def get_bfs(cell_ref,df):
    return concat_df(
        get_related_cells_subset(cell_ref,df)
    )
    
def query_df_with_rag(prompt, DF, top_n=5):
    # Calculate the top N similar results
    top_similar_results = calculate_similarity(prompt, DF, top_n=top_n)

    bfs_texts  = []
    scores = []
    for _, row in top_similar_results.iterrows():
        bfs_text = get_bfs(row['Cell Reference'], DF)
        bfs_texts.append(bfs_text)
        scores.append(row['similarity'])
    
    return bfs_texts,scores

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
    retrieved_info, scores = query_df_with_rag(prompt, DF, top_n=5)
    
    # Display retrieved information in the sidebar for verification
    st.sidebar.title("Retrieved Content for Verification")
    
    # Iterate over the rows and display the 'batch' content and similarity score
    for idx, (x, scr) in enumerate(zip(retrieved_info, scores)):
        #batch_text = row['batch']  # Access 'batch' column since it contains the data
        st.sidebar.markdown(f"**Batch {idx+1}:** {x}")
        st.sidebar.markdown(f"**Similarity:** {scr}")
        st.sidebar.markdown("---")

    # Combine the retrieved info into a string to be included in the assistant response
    combined_info = "\n\n".join(retrieved_info)

    assistant_response = ""
    if combined_info:
        #assistant_response += f"**Retrieved Info:** {combined_info}\n\n"
        st.session_state.messages.append({"role": "system", "content": f"**Retrieved Info:** {combined_info}\n\n"})
    
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