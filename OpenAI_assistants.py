"""
OpenAI Assistants using openai API (by T.-W. Yoon, Jan. 2024)
"""

import streamlit as st
import openai
import os, time, requests
import pickle
import hashlib
from io import BytesIO
from audio_recorder_streamlit import audio_recorder


def check_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.get(
        "https://api.openai.com/v1/engines", headers=headers
    )
    time.sleep(1)

    return response.status_code == 200


def run_thread(model, assistant_id, thread_id, query):
    """
    This function runs a conversation thread with an assistant.

    Args:
        assistant_id (str): The ID of the assistant.
        thread_id (str): The ID of the conversation thread.
        query (str): The user's query.

    Returns:
        message: Text of the message object in the conversation thread.
    """

    # Create the user message and add it to the thread
    st.session_state.client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=query,
    )

    # Create the Run, passing in the thread and the assistant
    run = st.session_state.client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        model=model
    )

    # Periodically retrieve the Run to check status and see if it has completed
    with st.spinner("AI is thinking..."):
        while run.status != "completed":
            time.sleep(1)
            run = st.session_state.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run.status in {"failed", "expired"}:
                st.error(f"The API request has {run.status}.", icon="🚨")
                return None

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=thread_id
    )
    message = messages.data[0].content[0].text

    return message


def process_citations(message):
    """
    This function processes citations in the given message,
    and returns the modified message, citations, and cited files.

    Args:
        message: The data of the original message object.

    Returns:
        tuple: A tuple containing the modified message content,
        citations (list), and cited files (list).
    """
    client = st.session_state.client

    annotations = message.annotations
    citations, cited_files = [], []

    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message.value = message.value.replace(
            annotation.text, f" [{index+1}]"
        )

        # Gather citations based on annotation attributes
        try:
            if (file_citation := getattr(annotation, 'file_citation', None)):
                cited_file = client.files.retrieve(file_citation.file_id)
            elif (file_path := getattr(annotation, 'file_path', None)):
                cited_file = client.files.retrieve(file_path.file_id)
            citations.append(file_citation.quote)
            cited_files.append(f"[{index+1}] :blue[{cited_file.filename}]")
        except Exception as e:
            # st.error(f"An error occurred: {e}", icon="🚨")
            # Ignore if there are problems with extracting citation information
            pass

    return message.value, citations, cited_files


def get_file_name(number, length=20):
    hashed = hashlib.sha256(str(number).encode('utf-8')).hexdigest()
    file_name = f"files/{hashed[:length]}.pickle"

    return file_name


def thread_exists(thread_id):
    """
    This function checks if the thread with a given id exists.
    """

    try:
        # Try to retrieve the thread
        st.session_state.client.beta.threads.retrieve(thread_id)
        return True
    except openai.APIError as e:
        # If the thread does not exist, return False
        return False


def show_thread_messages(thread_id):
    """
    This function shows all the messages of a given thread.
    """

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=thread_id
    )

    for message in reversed(messages.data):
        for role, content in zip(message.role, message.content):
            if role == "u":
                with st.chat_message("user"):
                    st.markdown(content.text.value)
            else:
                # Print the citation information
                content_text, citations, cited_files = process_citations(
                    content.text
                )
                with st.chat_message("assistant"):
                    st.markdown(content_text)
                    if citations:
                        with st.expander("Source(s)"):
                            for citation, file in zip(citations, cited_files):
                                st.markdown(file, help=citation)


def name_thread(thread_id):
    """
    This function a given thread a name with a summary of the first user query.
    """

    if not thread_exists(thread_id):
        return None

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=thread_id
    )

    first_query = messages.data[-1].content[0].text.value

    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant summarizing the user's "
                                + "message in 40 characters in noun form."
                },
                {
                    "role": "user",
                    "content": first_query
                }
            ],
            temperature=0.2,
        )
        thread_name = response.choices[0].message.content
    except openai.Error:
        thread_name = first_query[:40] + "..."

    return thread_name


def save_thread_info_file():
    with open(st.session_state.thread_info_pickle, 'wb') as file:
        pickle.dump(st.session_state.threads_list, file)
    time.sleep(1)


def load_thread_info_file():
    with open(st.session_state.thread_info_pickle, 'rb') as file:
        st.session_state.threads_list = pickle.load(file)
    st.session_state.thread_names = [
        thread["name"] for thread in st.session_state.threads_list 
    ]


def update_threads_info():
    st.session_state.thread_names = [
        thread["name"] for thread in st.session_state.threads_list 
    ]
    save_thread_info_file()


def create_new_thread():
    st.session_state.query_exists = False

    thread = st.session_state.client.beta.threads.create()
    st.session_state.threads_list.insert(0, {"id": thread.id, "name": "No name yet"})
    st.session_state.thread_index = 0

    update_threads_info()


def delete_thread(thread_index):
    if thread_exists(st.session_state.threads_list[thread_index]["id"]):
        st.session_state.client.beta.threads.delete(
            st.session_state.threads_list[thread_index]["id"]
        )

    del st.session_state.threads_list[thread_index]

    if st.session_state.threads_list:
        st.session_state.thread_index = 0
        update_threads_info()
    else:
        create_new_thread()


def upload_files():
    uploaded_files = st.file_uploader(
        label="Upload an article",
        type=["pdf"],
        accept_multiple_files=True,
        # on_change=reset_conversation,
        label_visibility="collapsed",
    )

    return uploaded_files


def upload_to_openai(filepath, purpose="assistants"):
    """
    This function uploads a file to OpenAI and returns its file ID.
    """

    with open(filepath, "rb") as file:
        response = openai.files.create(file=file.read(), purpose="assistants")
    return response.id


# def reset_conversation():
#     st.session_state.query_exists = False
#     if st.session_state.thread is not None:
#         delete_thread(st.session_state.thread.id)


def enable_user_input():
    st.session_state.query_exists = True


def read_audio(audio_bytes):
    """
    This function reads audio bytes and returns the corresponding text.
    """
    try:
        audio_data = BytesIO(audio_bytes)
        audio_data.name = "recorded_audio.wav"  # dummy name

        transcript = st.session_state.client.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )
        text = transcript.text
    except Exception as e:
        text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return text


def openai_assistants():
    """
    This main function presents OpenAI assistants by managing
    threads and messages.
    """

    st.write("## 📚 OpenAI Assistants")
    st.write("")

    with st.sidebar:
        st.write("")
        st.write("**Your API Key**")
        openai_api_key = st.text_input(
            label="$\\hspace{0.25em}\\texttt{Your OpenAI API Key}$",
            type="password",
            placeholder="sk-",
            label_visibility="collapsed",
        )

    if check_api_key(openai_api_key):
        st.session_state.openai_api_key = openai_api_key
        st.session_state.client = openai.OpenAI(api_key=openai_api_key)
    else:
        st.info(
            """
            **Enter your OpenAI API key in the sidebar**

            [Get an OpenAI API key](https://platform.openai.com/api-keys)
            The GPT-4 API can be accessed by those who have made
            a payment of $1 to OpenAI (a strange policy) at the time of
            writing this code.
            """
        )
        st.stop()

    if "thread_index" not in st.session_state:
        st.session_state.thread_index = 0

    if "threads_list" not in st.session_state:
        st.session_state.threads_list = []

    if "thread_names" not in st.session_state:
        st.session_state.thread_names = []

    if "query_exists" not in st.session_state:
        st.session_state.query_exists = False

    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    # Set the file name for storing thread information
    if "thread_info_pickle" not in st.session_state:
        st.session_state.thread_info_pickle = get_file_name(
            st.session_state.openai_api_key
        )

    if not os.path.exists(st.session_state.thread_info_pickle):
        # Save st.session_state.thread_dictionary to pickle
        create_new_thread()

    # Load st.session_state.thread_dictionary from pickle
    load_thread_info_file()

    # Find your assistants with the API key
    assistants = st.session_state.client.beta.assistants.list(
        order="desc",
        limit="20",
    )
    assistant_names = [assistant.name for assistant in assistants]

    with st.sidebar:
        st.write("")
        st.write("**Assistant(s)**")
        assistant = st.selectbox(
            label="Assistants",
            options=assistant_names,
            label_visibility="collapsed",
            index=0,
        )
        assistant_id = assistants.data[assistant_names.index(assistant)].id

        st.write("")
        st.write("**Models**")
        model = st.radio(
            label="Models",
            options=("gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
            label_visibility="collapsed",
            index=0,
        )

        st.write("")
        st.write("**Thread(s)**")
        thread_name = st.selectbox(
            label="Thread names",
            options=st.session_state.thread_names,
            label_visibility="collapsed",
            # on_change=change_thread,
        )
        st.write(
            "<small>Threads that have been inactive for 60 days will be deleted.</small>",
            unsafe_allow_html=True,
        )

        thread_index = st.session_state.thread_names.index(thread_name)
        if thread_exists(st.session_state.threads_list[thread_index]["id"]):
            st.session_state.thread_index = thread_index
        else:
            delete_thread(thread_index)
            st.rerun()

        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Aug. 2023  \n</small>",
            "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
            "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
            unsafe_allow_html=True,
        )
        # if st.button("Finish"):
        #     os._exit(0)

    st.write("##### Conversation with Assistant")
    st.write("")

    if st.session_state.threads_list:
        thread_id = st.session_state.threads_list[st.session_state.thread_index]["id"]
        show_thread_messages(thread_id)

    # Reset the conversation
    left, right = st.columns(2)
    left.button(
        label="$~\:\:\,$Create a new thread$~\:\:\:$",
        on_click=create_new_thread,
    )
    right.button(
        label="Delete the current thread",
        on_click=delete_thread,
        args=(st.session_state.thread_index,)
    )

    # Use your keyboard
    user_input = st.chat_input(
        placeholder="Enter your query",
        on_submit=enable_user_input,
    )

    # Use your microphone
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes != st.session_state.audio_bytes:
        query = read_audio(audio_bytes)
        st.session_state.audio_bytes = audio_bytes
        if query is not None:
            st.session_state.query_exists = True
    elif user_input and st.session_state.query_exists:
        query = user_input.strip()

    if st.session_state.query_exists:
        with st.chat_message("user"):
            st.markdown(query)
        run_thread(model, assistant_id, thread_id, query)

        if st.session_state.threads_list[thread_index]["name"] == "No name yet":
            thread_name = name_thread(thread_id)
            st.session_state.threads_list[thread_index]["name"] = thread_name
            update_threads_info()

        st.session_state.query_exists = False
        st.rerun()


if __name__ == "__main__":
    openai_assistants()
