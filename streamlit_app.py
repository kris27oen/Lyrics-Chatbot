from openai import OpenAI
import streamlit as st
import re
import time
import requests

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


def main():
    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
            },
        page_icon="img/favicon.ico"
    )

    if 'song_info' not in st.session_state:
        st.session_state.song_info = None  # Initialize with None
    st.title(f"💬 {user_name}'s MusicDJ")
    st.markdown("Wussup!!👋 I'm MusixDJ and I'll be assisting you with your music taste 🎧 \n\n⭐As a starter, you can ask me any song lyrics by entering --- [Lyrics 'song_name' by artist_name]")

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

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"], avatar=user_image).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"], avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

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
        else:
            st.session_state.song_info = None
            return "Please follow the instruction by using the provided command!"


    def chat(prompt: str):
        st_c_chat.chat_message("user", avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)


if __name__ == "__main__":
    main()
