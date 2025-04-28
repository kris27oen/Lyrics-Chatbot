from openai import OpenAI
import streamlit as st
import re
import numpy as np
import time
import requests
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords

placeholderstr = "Please input your command"
user_name = "Owen"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

########################################################
## Spotify Authentication
CLIENT_ID = "b9e0979d54c449d4a1b7f23a1be1d329"
CLIENT_SECRET = "03559d2dc6b643e8af412d5930ee4ec2"

auth_url = "https://accounts.spotify.com/api/token"
########################################################

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)


# Function to generate 2D plot for
def generate_2d_plot(prompt: str):
    # Preprocess the input text to tokenize the words
    tokenized_sentence = simple_preprocess(prompt)
    
    # Train Word2Vec model (this is just an example, you may want to load a pre-trained model instead)
    model = Word2Vec([tokenized_sentence], vector_size=100, window=5, min_count=1, workers=4)
    
    # Extract word vectors
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    
    # Reduce the dimensions to 2D using PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)
    
    # Create a scatter plot for the 2D visualization
    word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]
    scatter = go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color='blue', size=8),
        customdata=['word' for _ in model.wv.index_to_key],  # Optional: Add custom data to be displayed on hover
        ids=word_ids,
        hovertemplate="Word: %{text}<br>Color: %{customdata}"
    )

    # Create line traces for each word in the sentence (showing the order of words)
    line_traces = []
    for i, word in enumerate(model.wv.index_to_key):
        line_trace = go.Scatter(
            x=[reduced_vectors[i, 0]],
            y=[reduced_vectors[i, 1]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            name=f"{word}",
            text=[word],
            hoverinfo='text'
        )
        line_traces.append(line_trace)

    # Combine scatter and line traces
    fig = go.Figure(data=[scatter] + line_traces)

    # Set the plot title and axis labels
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title="2D Visualization of Word Embeddings",
        width=1000,  # Custom width
        height=1000  # Custom height
    )

    # Show the plot in the Streamlit app
    st.plotly_chart(fig)


# Function to generate 3D plot
def generate_3d_plot(prompt: str):
     # Preprocess the input text to tokenize the words
    tokenized_sentence = simple_preprocess(prompt)
    
    # Train Word2Vec model (this is just an example, you may want to load a pre-trained model instead)
    model = Word2Vec([tokenized_sentence], vector_size=100, window=5, min_count=1, workers=4)
    
    # Extract word vectors
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    
    # Reduce the dimensions to 3D using PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(word_vectors)
    
    # Create a 3D scatter plot for the visualization
    word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]
    scatter = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color='blue', size=8),
        customdata=['word' for _ in model.wv.index_to_key],  # Optional: Add custom data to be displayed on hover
        ids=word_ids,
        hovertemplate="Word: %{text}<br>Color: %{customdata}"
    )

    # Create line traces for each word in the sentence (showing the order of words)
    line_traces = []
    for i, word in enumerate(model.wv.index_to_key):
        line_trace = go.Scatter3d(
            x=[reduced_vectors[i, 0]],
            y=[reduced_vectors[i, 1]],
            z=[reduced_vectors[i, 2]],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='circle'),
            name=f"{word}",
            text=[word],
            textfont=dict(color='black'),  # Set text color to white
            hoverinfo='text'
        )
        line_traces.append(line_trace)

    # Combine scatter and line traces
    fig = go.Figure(data=[scatter] + line_traces)

    # Set the plot title and axis labels
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Visualization of Word Embeddings",
        width=1000,  # Custom width
        height=1000  # Custom height
    )

    # Show the plot in the Streamlit app
    st.plotly_chart(fig)


# Function to handle Q2 (text-based analysis)
def similarities_analysis(song, keyword, flag=1):
     # Preprocess the input prompt (remove stopwords, tokenize)
    tokenized_words = [simple_preprocess(remove_stopwords(song))]
    # print("keyword:", keyword)
    # print("Tokenized sentences:", tokenized_words)
    # Initialize the Word2Vec model with Skip-gram (sg=1)
    vector_size = 10
    window_size = 4
    min_count = 1
    workers = 4
    sg_flag = flag  # Skip-gram model

    # Train the Word2Vec Skip-gram model
    model = Word2Vec(tokenized_words, vector_size=vector_size, window=window_size, min_count=min_count, workers=workers, sg=sg_flag)
    # print(model.wv[keyword])
    # Check if the keyword exists in the vocabulary
    if keyword in model.wv:
        # Get the most similar words to the given keyword
        similar_words = model.wv.most_similar(keyword)
        answer = f"Most similar words to '{keyword}': {similar_words[0]}"
    else:
        answer = f"Keyword '{keyword}' not found in the model's vocabulary."
    
    return answer


# Function to handle navigation between pages
def handle_navigation(page_selected):
    if page_selected == "Home":
        st.title(f"💬 {user_name}'s MusicDJ")
        st.markdown("Wussup!!👋 I'm MusixDJ and I'll be assisting you with your music taste 🎧 \n\n⭐As a starter, you can ask me any song lyrics by entering --- [Lyrics 'song_name' by artist_name]")
    
    elif page_selected == "Q1: 2D & 3D Word Embeddings":
        st.title("Q1: 2D & 3D Word Embedding Visualizations")
        st.markdown("Here you can visualize word embeddings in 2D and 3D. But before that, please ask me for the lyrics of a song by entering --- [Lyrics 'song_name' by artist_name]")
        st.markdown("You can also generate 2D and 3D plots by entering --- [Generate 2D plot] or [Generate 3D plot]")
        
    elif page_selected == "Q2: Similarities using Word2Vec (Skip-gram)":
        st.title("Q2: Similarities using Word2Vec (Skip-gram)")
        st.markdown("Here you can analyze the similarities of words using the Skip-gram model. But before that, please ask me for the lyrics of a song by entering --- [Lyrics 'song_name' by artist_name]")
        st.markdown("You can also analyze the similarities by entering --- [Similarities 'keyword']")
        st.markdown("Note: The keyword should be a word from the lyrics.")

    elif page_selected == "Q3: Similarities using Word2Vec (CBOW)":
        st.title("Q3: Similarities using Word2Vec (CBOW)")
        st.markdown("Here you can analyze the similarities of words using the CBOW model. But before that, please ask me for the lyrics of a song by entering --- [Lyrics 'song_name' by artist_name]")
        st.markdown("You can also analyze the similarities by entering --- [Similarities 'keyword']")
        st.markdown("Note: The keyword should be a word from the lyrics.")

    else:
        st.write("Select a valid page.")


def main():
    st.set_page_config(
        page_title='MusicDJ',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
            },
        page_icon="img/favicon.ico"
    )
    
    # Sidebar for page navigation
    pages = ["Home", "Q1: 2D & 3D Word Embeddings", "Q2: Similarities using Word2Vec (Skip-gram)", "Q3: Similarities using Word2Vec (CBOW)"]
    page_selected = st.sidebar.selectbox("Choose a Page", pages)

     # Display the selected page content
    handle_navigation(page_selected)

    if 'song_info' not in st.session_state:
        st.session_state.song_info = None  # Initialize with None
    if page_selected not in st.session_state or not isinstance(st.session_state[page_selected], list):
        st.session_state[page_selected] = []  # Initialize as a list to store page-specific chat histories


    with st.sidebar:
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=1)
        if 'lang_setting' in st.session_state:
            lang_setting = st.session_state['lang_setting']
        else:
            lang_setting = selected_lang
            st.session_state['lang_setting'] = lang_setting

        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image("https://www.w3schools.com/howto/img_avatar.png")
    

    st_c_chat = st.container(border=True)

    # Create or update chat history based on selected page
    def update_chat_history(page_selected, prompt, response):
        if page_selected not in st.session_state:
            st.session_state[page_selected] = []
            
        st_c_chat.chat_message("user", avatar=user_image).write(prompt)
        st.session_state[page_selected].append({"role": "user", "content": prompt})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))
        st.session_state[page_selected].append({"role": "assistant", "content": response})

    # Display chat history for the selected page
    def display_chat_history(page_selected):
        if page_selected in st.session_state:
            for msg in st.session_state[page_selected]:
                if msg["role"] == "user":
                    st_c_chat.chat_message(msg["role"], avatar=user_image).markdown(msg["content"])
                elif msg["role"] == "assistant":
                    st_c_chat.chat_message(msg["role"]).markdown(msg["content"])
                    

    def search(title, artist=None):
        # Get the access token for Spotify authentication
        auth_response = requests.post(
            auth_url,
            data={"grant_type": "client_credentials"},
            auth=(CLIENT_ID, CLIENT_SECRET)
        )
        access_token = auth_response.json().get("access_token")
        if not access_token:
            raise Exception("Failed to get Spotify access token")
        
        query = title
        if artist: 
            query += " artist:" + artist
        
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "q": query,  # Search query
            "type": "track",
            "limit": 1  # Max per request
        }
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params=params
        )
        
        # Extract the track information
        tracks = response.json().get("tracks", {}).get("items", [])
        if not tracks:
            return "No tracks found for this title."
        
        first_track = tracks[0]  # Get the first track
        track_name = first_track["name"]
        artist_name = first_track["artists"][0]["name"]
        release_date = first_track["album"]["release_date"]
        
        return track_name, artist_name, release_date


    def get_lyrics(title, artist):
        # Format artist/title to URL-safe strings
        formatted_artist = artist.strip().lower().replace(" ", "%20")
        formatted_title = title.strip().lower().replace(" ", "%20")

        url = f"https://api.lyrics.ovh/v1/{formatted_artist}/{formatted_title}"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json().get("lyrics")
        else:
            return None  # Lyrics not found


    def extract_song_and_artist(prompt):
        # Define a regex pattern to match 'Song Title' by Artist format
        pattern = r"['\"](.*?)['\"]\s+by\s+([a-zA-Z\s]+)"
        
        # Search for the pattern in the prompt
        match = re.search(pattern, prompt, re.IGNORECASE)
        
        if match:
            song_title = match.group(1)
            artist_name = match.group(2)
            return song_title, artist_name
        else:
            return None, None


    def generate_response(prompt):
        if 'lyrics' in prompt.lower():
            # Try to extract the song title and artist from the prompt
            title_search, artist_search = extract_song_and_artist(prompt)

            # If no title or artist is extracted, return an error message
            if not title_search or not artist_search:
                return "Sorry, I couldn't extract the song title and artist from your prompt."

            # Search for the song using the extracted title and artist
            song_name, artist_name, release_date = search(title_search, artist_search) 
            
            if song_name and artist_name:
                # Fetch the lyrics for the song
                lyrics = get_lyrics(song_name, artist_name)
                if lyrics:
                    st.session_state.song_info = {
                        "song_name": song_name,
                        "artist_name": artist_name,
                        "lyrics": lyrics
                    }
                    return f"Lyrics of {song_name} by {artist_name}:\n\n{lyrics[:500]}... \n\nFor the full lyrics, Enter \"Full Song\"."  # Truncate lyrics for brevity
                else:
                    return f"Sorry, I couldn't find the lyrics for the song. {title_search}, {artist_search}"
            else:
                return "Sorry, I couldn't find the song based on the provided title and artist."
        
        elif 'full song' in prompt.lower():
            # Retrieve the stored song and lyrics from session state
            if st.session_state.song_info:
                song_info = st.session_state.song_info
                st.session_state.song_info = None
                return f"Full lyrics of {song_info['song_name']} by {song_info['artist_name']}:\n\n{song_info['lyrics']}"
            else:
                return "I couldn't find any lyrics. Please ask for the lyrics first."
        
        elif page_selected == "Q1: 2D & 3D Word Embeddings" and "generate 2d plot" in prompt.lower():
            song_info = st.session_state.song_info
            if song_info["song_name"] is not None:  
                generate_2d_plot(song_info["lyrics"])
                return f"2D plot of \"{song_info['song_name']}\" generated."
            else:
                return "Please ask for the lyrics first."
        
        elif page_selected == "Q1: 2D & 3D Word Embeddings" and "generate 3d plot" in prompt.lower():
            song_info = st.session_state.song_info
            if song_info["song_name"] is not None:  
                generate_3d_plot(song_info["lyrics"])
                return f"3D plot of \"{song_info['song_name']}\" generated."
            else:
                return "Please ask for the lyrics first."
        
        elif page_selected == "Q2: Similarities using Word2Vec (Skip-gram)" and "similarities" in prompt.lower():
            song_info = st.session_state.song_info
            if song_info["song_name"] is not None:  
                keyword = re.search(r"similarities\s*['\"](.*?)['\"]", prompt, re.IGNORECASE)
                keyword = keyword.group(1).strip().lower() if keyword else ""
                answer = similarities_analysis(song_info["lyrics"], keyword, flag=1)
                return answer
            else:
                return "Please ask for the lyrics first."
        
        elif page_selected == "Q3: Similarities using Word2Vec (CBOW)" and "similarities" in prompt.lower():
            song_info = st.session_state.song_info
            if song_info["song_name"] is not None:  
                keyword = re.search(r"similarities\s*['\"](.*?)['\"]", prompt, re.IGNORECASE)
                keyword = keyword.group(1).strip().lower() if keyword else ""
                answer = similarities_analysis(song_info["lyrics"], keyword, flag=0)
                return answer
            else:
                return "Please ask for the lyrics first."
        
        else:
            st.session_state.song_info = None
            return "Please follow the instruction by using the provided command!"

    # Display chat history for the selected page
    display_chat_history(page_selected)

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        response = generate_response(prompt)
        update_chat_history(page_selected, prompt, response)


if __name__ == "__main__":
    main()
