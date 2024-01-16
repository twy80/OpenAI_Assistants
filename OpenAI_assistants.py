"""
OpenAI Assistants using openai API (by T.-W. Yoon, Jan. 2024)
"""

import streamlit as st
import os
import time
import requests
import pickle
import hashlib
from openai import OpenAI, APIError
from io import BytesIO
from audio_recorder_streamlit import audio_recorder


def check_api_key(api_key):
    """
    Return True if the given OpenAI api_key is valid.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.get(
        "https://api.openai.com/v1/engines", headers=headers
    )

    return response.status_code == 200


def run_thread(model, assistant_id, thread_id, query, file_ids):
    """
    Run a conversation thread with an assistant.

    Args:
        model (str): The GPT model used.
        assistant_id (str): The ID of the assistant.
        thread_id (str): The ID of the conversation thread.
        query (str): The user's query.
        file_ids (list of strings): list of file IDs.

    Returns:
        message: Text of the message object in the conversation thread.
    """

    try:
        # Create the user message and add it to the thread
        st.session_state.client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=query, file_ids=file_ids
        )

        # Create the Run, passing in the thread and the assistant
        run = st.session_state.client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_id, model=model
        )
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        return None

    # Periodically retrieve the Run to check status and see if it has completed
    with st.spinner("AI is thinking..."):
        while run.status != "completed":
            time.sleep(1)
            run = st.session_state.client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run.id
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
    Process citations in the given message,
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
        message.value = message.value.replace(annotation.text, f" [{index+1}]")

        # Gather citations based on annotation attributes
        try:
            if (file_citation := getattr(annotation, "file_citation", None)):
                cited_file = client.files.retrieve(file_citation.file_id)
            elif (file_path := getattr(annotation, "file_path", None)):
                cited_file = client.files.retrieve(file_path.file_id)
            citations.append(file_citation.quote)
            cited_files.append(f"[{index+1}] :blue[{cited_file.filename}]")
        except Exception as e:
            # st.error(f"An error occurred: {e}", icon="🚨")
            # Ignore if there are problems with extracting citation information
            pass

    return message.value, citations, cited_files


def get_file_path(number, length=20):
    """
    Return a file path of a given length using the hashed value of number.
    """

    hashed = hashlib.sha256(str(number).encode("utf-8")).hexdigest()
    file_name = f"files/{hashed[:length]}.pickle"

    return file_name


def thread_exists(thread_id):
    """
    Check to see if the thread with a given id exists.
    """

    try:
        # Try to retrieve the thread
        st.session_state.client.beta.threads.retrieve(thread_id)
        return True
    except APIError:
        # If the thread does not exist, return False
        return False


def show_thread_messages(thread_id, no_of_messages="All"):
    """
    Show the most recent 'no_of_messages' messages of a given thread.
    If 'no_of_messages' is None, all the messages are shown.
    """

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=thread_id
    )

    if no_of_messages == "All":
        messages = messages.data
    elif isinstance(no_of_messages, int) and no_of_messages > 0:
        messages = messages.data[:no_of_messages]
    else:
        st.error("'no_of_messages' is a positive integer or 'All'", icon="🚨")

    for message in reversed(messages):
        for role, content in zip(message.role, message.content):
            if role == "u":
                with st.chat_message("user"):
                    st.markdown(content.text.value)
            elif hasattr(content, "text"):
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
    Name the thread with the given ID using a summary of the first user query.
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
                    + "message in 40 characters in noun form.",
                },
                {"role": "user", "content": first_query},
            ],
            temperature=0.2,
        )
        thread_name = response.choices[0].message.content
    except APIError:
        thread_name = first_query[:40] + "..."

    return thread_name


def save_thread_info_file():
    """
    Save a list containing the thread ids, names and a list of file ids
    used in the thread to a pickle file.
    """

    with open(st.session_state.thread_info_pickle, "wb") as file:
        pickle.dump(st.session_state.threads_list, file)
    time.sleep(1)


def load_thread_info_file():
    """
    Load a list containing the thread ids, names and a list of file ids
    from a pickle file, and reset the list of thread names.
    """

    with open(st.session_state.thread_info_pickle, "rb") as file:
        st.session_state.threads_list = pickle.load(file)
    st.session_state.thread_names = [
        thread["name"] for thread in st.session_state.threads_list
    ]


def update_threads_info():
    """
    Reset the list of thread names, and save a list containing the thread
    ids, names and a list of file ids to a pickle file.
    """

    st.session_state.thread_names = [
        thread["name"] for thread in st.session_state.threads_list
    ]
    save_thread_info_file()


def create_new_thread():
    """
    Create a new thread, and add it to the top of the threads list containing
    the thread ids, names and a list of file ids. The name of this new thread
    is "No name yet".
    """

    thread = st.session_state.client.beta.threads.create()
    st.session_state.threads_list.insert(
        0, {"id": thread.id, "name": "No name yet", "file_ids": []}
    )
    st.session_state.thread_index = 0

    update_threads_info()


def delete_file(file_id):
    """
    Delete the file of the given id.
    """

    client = st.session_state.client
    assistants = st.session_state.client.beta.assistants.list(
        order="desc",
        limit="20",
    ).data

    if assistants:
        for assistant in assistants:
            for id in assistant.file_ids:
                if id == file_id:
                    try:
                        # Delete the association with assistants
                        client.beta.assistants.files.delete(
                            assistant_id = assistant.id,
                            file_id=file_id
                        )
                    except Exception as e:
                        # st.error(f"An error occurred: {e}", icon="🚨")
                        pass

    try:
        client.files.delete(file_id)
    except Exception as e:
        # st.error(f"An error occurred: {e}", icon="🚨")
        pass


def delete_thread(thread_index):
    """
    Delete a thread and remove the corresponding element from the list
    containing the thread ids, names and a list of file ids.
    All the files used in the thread are deleted as well.
    """

    if thread_exists(st.session_state.threads_list[thread_index]["id"]):
        for file_id in st.session_state.threads_list[thread_index]["file_ids"]:
            delete_file(file_id)
        st.session_state.client.beta.threads.delete(
            st.session_state.threads_list[thread_index]["id"]
        )

    del st.session_state.threads_list[thread_index]

    if st.session_state.threads_list:
        st.session_state.thread_index = 0
        update_threads_info()
    else:
        create_new_thread()


def upload_to_openai(file_path, purpose="assistants"):
    """
    Upload a file to OpenAI and return its file ID.
    Return None if there are errors.
    """

    try:
        with open(file_path, "rb") as file:
            upload_file = st.session_state.client.files.create(
                file=file, purpose=purpose
            )
        return upload_file.id

    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        return None


def upload_files(type=None):
    """
    Upload files and return the list of the uploaded file ids.
    If no files are uploaded, return an empty list.
    """

    uploaded_files = st.file_uploader(
        label="Upload an article",
        type=type,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=st.session_state.uploader_key,
    )

    if uploaded_files:
        uploaded_file_ids = []
        for uploaded_file in uploaded_files:
            with open(f"{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_id = upload_to_openai(f"{uploaded_file.name}")
            if file_id is not None:
                uploaded_file_ids.append(file_id)
        return uploaded_file_ids
    else:
        return []


def make_unique_names(list_of_names):
    # Make all elements in list_of_names unique
    counts = {}
    list_of_unique_names = []

    for name in list_of_names:
        if name in counts:
            counts[name] += 1
            name = f"{name} ({counts[name]})"
        else:
            counts[name] = 0
        list_of_unique_names.append(name)

    return list_of_unique_names


def show_files():
    st.write("")
    st.write("**All file(s)** $\,$(uploaded to OpenAI)")

    files = st.session_state.client.files.list().data
    if files:
        file_names = [file.filename for file in files]
        file_names = make_unique_names(file_names)
        file_name = st.selectbox(
            label="File names",
            options=file_names,
            label_visibility="collapsed",
            index=0,
        )
        index = file_names.index(file_name)
        file = files[index]
        st.write(
            f"""
            - :blue[File Name]: {file.filename}
            - :blue[File ID]: {file.id}
            - :blue[Purpose]: {file.purpose}
            """
        )

        st.button(
            label="Delete the file",
            on_click=delete_file,
            args=(file.id,),
        )
    else:
        st.write("No uploaded file")


def set_assistants_list():
    """
    Set the session state dictionary containing assistant names and ids.
    """

    assistants = st.session_state.client.beta.assistants.list(
        order="desc",
        limit="20",
    ).data
    if assistants:
        st.session_state.assistants_name_id = [
            (assistant.name, assistant.id) for assistant in assistants
        ]
    else:
        st.session_state.assistants_name_id = []
        st.session_state.run_assistants = False


def delete_assistant(assistant_id):
    """
    Delete the assistant of the given id along with the associated files.
    """

    assistant = st.session_state.client.beta.assistants.retrieve(assistant_id)
    for file_id in assistant.file_ids:
        try:
            st.session_state.client.files.delete(file_id)
        except Exception as e:
            # st.error(f"An error occurred: {e}", icon="🚨")
            pass
    try:
        st.session_state.client.beta.assistants.delete(assistant_id)
    except Exception as e:
        # st.error(f"An error occurred: {e}", icon="🚨")
        pass


def run_or_manage_assistants():
    # Toggle the flag to determine whether to run or manage assistants
    st.session_state.run_assistants = not st.session_state.run_assistants


def read_audio(audio_bytes):
    """
    Read audio bytes and return the corresponding text.
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
        # st.error(f"An error occurred: {e}", icon="🚨")

    return text


def show_assistant(assistant_id):
    """
    Show the information of an assistant object
    """

    assistant = st.session_state.client.beta.assistants.retrieve(assistant_id)
    tools = [tool.type for tool in assistant.tools]
    file_ids = [id[0:12] + "..." for id in assistant.file_ids]

    st.write("")
    if st.button(label="Create an assistant"):
        st.session_state.manage_assistant_app = "create"
        st.rerun()

    st.write("")
    st.write("**Assistant**")

    if assistant_id is not None:
        st.write(
            f"""
            - :blue[Name]: {assistant.name}
            - :blue[Default Model]: {assistant.model}
            - :blue[ID]: {assistant.id}
            - :blue[Instructions]: {assistant.instructions}
            - :blue[Description]: {assistant.description}
            - :blue[Tool(s)]: {", ".join(tools)}
            - :blue[File ID(s)]: {", ".join(file_ids)}
            """
        )
        left, right = st.columns(2)
        if left.button(label="Modify the assistant"):
            st.session_state.manage_assistant_app = "modify"
            st.rerun()
        if right.button(label="Delete the assistant"):
            delete_assistant(assistant_id)
            set_assistants_list()
            st.rerun()
    else:
        st.write("No assistant yet")

    show_files()


def update_assistant(assistant_id):
    """
    Update the assistant with 'assistant_id', or
    create an assistant when 'assistant_id' is None
    """

    model_options = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
    if assistant_id is None:
        st.write("**:blue[Create your assistant]**")
        model_index = 0
        assistant_name_value = ""
        instructions_value = ""
        description_value = ""
        tools_value = None
    else:
        assistant = st.session_state.client.beta.assistants.retrieve(
            assistant_id
        )
        st.write(f"**:blue[Modify the assistant] $\,${assistant.name}**")
        model_index = model_options.index(assistant.model)
        assistant_name_value = assistant.name
        instructions_value = assistant.instructions
        description_value = assistant.description
        tools_value = [tool.type for tool in assistant.tools]
        file_ids_value = assistant.file_ids

    with st.form("Submit"):
        st.write(
            """
            **Model** $\,$(This default model will be overriden
            by the model selected at the time of running threads.)
            """
        )
        model = st.radio(
            label="Default models",
            options=("gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
            label_visibility="collapsed",
            index=model_index,
        )
        st.write("**Name**")
        name = st.text_input(
            label="assistant name",
            value=assistant_name_value,
            label_visibility="collapsed",
        )
        st.write("**Instructions**")
        instructions = st.text_area(
            label="instructions",
            value=instructions_value,
            label_visibility="collapsed",
        )
        st.write("**Description**")
        description = st.text_area(
            label="description",
            value=description_value,
            label_visibility="collapsed",
        )
        st.write("**Tools** ('function' has not been implemented yet.)")
        tool_options = ["retrieval", "code_interpreter"]
        tool_names = st.multiselect(
            label="assistant tools",
            options=tool_options,
            default=tools_value,
            label_visibility="collapsed",
        )
        tools = [{"type": tool_name} for tool_name in tool_names]
        st.write("**File(s)**$\,$ (to be added)")
        file_ids = upload_files()
        if assistant_id:
            file_ids.extend(file_ids_value)

        form_left, form_right = st.columns(2)
        submitted = form_left.form_submit_button("Submit")
        if submitted:
            if assistant_id is None:
                st.session_state.client.beta.assistants.create(
                    model=model,
                    name=name,
                    instructions=instructions,
                    description=description,
                    tools=tools,
                    file_ids=file_ids,
                )
            else:
                st.session_state.client.beta.assistants.update(
                    assistant_id=assistant_id,
                    model=model,
                    name=name,
                    instructions=instructions,
                    description=description,
                    tools=tools,
                    file_ids=file_ids,
                )
            set_assistants_list()
            st.session_state.manage_assistant_app = "show"
            st.rerun()

        back_to_manage = form_right.form_submit_button("Back")
        if back_to_manage:
            st.session_state.manage_assistant_app = "show"
            st.rerun()


def manage_assistant(assistant_id):
    """
    Create or modify assistants.
    """

    st.write("##### Managing Assistants")

    if st.session_state.manage_assistant_app == "show":
        func = update_assistant if assistant_id is None else show_assistant
        func(assistant_id)
    elif st.session_state.manage_assistant_app == "modify":
        update_assistant(assistant_id)
    else:
        update_assistant(None)


def run_assistant(model, assistant_id, thread_index):
    """
    Run a conversation with the specified assistant.

    Args:
        model (str): The model used for the assistant.
        assistant_id (str): The ID of the assistant.
        thread_index (int): The index of the thread to interact with.

    Returns:
        None
    """

    st.write("##### Conversation with Assistant")
    st.write("")

    thread_id = st.session_state.threads_list[thread_index]["id"]

    if st.session_state.threads_list:
        show_thread_messages(thread_id, st.session_state.no_of_messages)

    query = st.chat_input(
        placeholder="Enter your query",
    )

    if query or st.session_state.text_from_audio:
        if st.session_state.text_from_audio:
            query = st.session_state.text_from_audio
            st.session_state.text_from_audio = None

        with st.chat_message("user"):
            st.markdown(query)

        # Append the file ids in this message to st.session_state.threads_list
        if st.session_state.file_ids:
            st.session_state.threads_list[thread_index]["file_ids"].extend(
                st.session_state.file_ids
            )
            update_threads_info()
        try:
            message = run_thread(
                model, assistant_id, thread_id, query, st.session_state.file_ids
            )
        except Exception as e:
            message = None
            st.error(f"An error occurred: {e}", icon="🚨")

        st.session_state.file_ids = []
        st.session_state.uploader_key += 1

        if message is None:
            st.error("Request not completed.", icon="🚨")
        else:
            # Print the citation information
            content_text, citations, cited_files = process_citations(message)
            with st.chat_message("assistant"):
                st.markdown(content_text)
                if citations:
                    with st.expander("Source(s)"):
                        for citation, file in zip(citations, cited_files):
                            st.markdown(file, help=citation)

                assistant = st.session_state.client.beta.assistants.retrieve(assistant_id)
                st.markdown(
                    f"<small>(:blue[Assistant]: {assistant.name}, </small>"
                    + f"<small>:blue[Model] = {model})</small>",
                    unsafe_allow_html=True,
                )

            if st.session_state.threads_list[thread_index]["name"] == "No name yet":
                thread_name = name_thread(thread_id)
                st.session_state.threads_list[thread_index]["name"] = thread_name
                update_threads_info()

    # st.session_state.file_ids = upload_files(["pdf", "txt"])
    st.session_state.file_ids = upload_files()
    st.markdown(
        "<small>If you press :blue[Delete this thread] in the sidebar, </small>"
        + "<small>all the files used in the thread will be deleted </small>"
        + "<small>together with the thread.</small>",
        unsafe_allow_html=True,
    )

    # Use your microphone
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes != st.session_state.audio_bytes:
        st.session_state.text_from_audio = read_audio(audio_bytes)
        st.session_state.audio_bytes = audio_bytes
        if st.session_state.text_from_audio is not None:
            st.rerun()


def openai_assistants():
    """
    This main function presents OpenAI assistants by managing assistants,
    threads, messages, and files.
    """

    st.write("## 📚 OpenAI Assistants")
    st.write("")

    with st.sidebar:
        st.write("")
        st.write("**API Key Selection**")
        choice_api = st.sidebar.radio(
            label="$\\textsf{API Key Selection}$",
            options=("Your key", "My key"),
            label_visibility="collapsed",
            horizontal=True,
        )
        if choice_api == "Your key":
            st.write("**Your API Key**")
            st.session_state.openai_api_key = st.text_input(
                label="$\\textsf{Your API Key}$",
                type="password",
                placeholder="sk-",
                label_visibility="collapsed",
            )
            authentication = True
        else:
            st.session_state.openai_api_key = st.secrets["openai_api_key"]
            stored_pin = st.secrets["user_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="$\\textsf{Password}$",
                type="password",
                label_visibility="collapsed",
            )
            authentication = user_pin == stored_pin

    if authentication:
        if check_api_key(st.session_state.openai_api_key):
            st.session_state.client = OpenAI(
                api_key=st.session_state.openai_api_key
            )
            # Set the variable st.session_state.assistants_name_id
            # containing assistant names and ids
            set_assistants_list()
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
    else:
        st.info("**Enter the correct password in the sidebar**")
        st.stop()

    if "thread_index" not in st.session_state:
        st.session_state.thread_index = 0

    if "threads_list" not in st.session_state:
        st.session_state.threads_list = []

    if "thread_names" not in st.session_state:
        st.session_state.thread_names = []

    if "no_of_messages" not in st.session_state:
        st.session_state.no_of_messages = "All"

    if "file_ids" not in st.session_state:
        st.session_state.file_ids = []

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # Set the file name for storing thread information
    if "thread_info_pickle" not in st.session_state:
        st.session_state.thread_info_pickle = get_file_path(
            st.session_state.openai_api_key
        )

    if "run_assistants" not in st.session_state:
        st.session_state.run_assistants = True

    if "assistant_index" not in st.session_state:
        st.session_state.assistant_index = 0

    if "assistants_name_id" not in st.session_state:
        st.session_state.assistants_name_id = []

    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    if "text_from_audio" not in st.session_state:
        st.session_state.text_from_audio = None

    # Choose app for manage_assistants()
    if "manage_assistant_app" not in st.session_state:
        st.session_state.manage_assistant_app = "show"

    if not os.path.exists(st.session_state.thread_info_pickle):
        # Create an empty pickle and a thread
        with open(st.session_state.thread_info_pickle, 'w') as file:
            pass
        create_new_thread()

    # Load st.session_state.thread_dictionary from pickle
    load_thread_info_file()

    with st.sidebar:
        st.write("**Model**")
        model = st.radio(
            label="$\\textsf{Models}$",
            options=("gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
            label_visibility="collapsed",
            index=0,
        )

        # Make a list of assistant names
        assistant_names = [
            name for (name, _) in st.session_state.assistants_name_id
        ]
        assistant_names = make_unique_names(assistant_names)
        st.write("**Assistant(s)**")
        if assistant_names:
            if st.session_state.assistant_index >= len(assistant_names):
                st.session_state.assistant_index = 0
            assistant_name = st.selectbox(
                label="$\\textsf{Assistant(s)}$",
                options=assistant_names,
                label_visibility="collapsed",
                index=st.session_state.assistant_index,
            )
            index = assistant_names.index(assistant_name)
            assistant_id = st.session_state.assistants_name_id[index][1]
            st.session_state.assistant_index = index

            if st.session_state.run_assistants:
                run_or_manage = "$\;$Manage assistants$\;$"
            else:
                run_or_manage = "$~~\:\:$Run assistants$~~\:\:$"
            st.button(
                label=run_or_manage,
                on_click=run_or_manage_assistants,
                key="run_or_manage",
            )
        else:
            st.write("No assistant yet")
            st.session_state.run_assistants = False
            assistant_id = None

        st.write("**Thread(s)**")
        thread_name = st.selectbox(
            label="$\\textsf{Thread(s)}$",
            options=st.session_state.thread_names,
            label_visibility="collapsed",
        )
        st.write(
            "<small>Threads that have been inactive for 60 days will be deleted.</small>",
            unsafe_allow_html=True,
        )

        thread_index = st.session_state.thread_names.index(thread_name)
        thread_id = st.session_state.threads_list[thread_index]["id"]
        if thread_exists(thread_id):
            st.session_state.thread_index = thread_index
        else:
            delete_thread(thread_index)
            st.rerun()

        st.write("**Messages to show**")
        st.session_state.no_of_messages = st.radio(
            label="$\\textsf{Messages to show}$",
            options=("All", 10, 6),
            label_visibility="collapsed",
            horizontal=True,
            index=1,
        )

        st.write("")
        st.button(
            label="Create a new thread",
            on_click=create_new_thread,
        )
        st.button(
            label="$\:\,$Delete this thread$\:\,$",
            on_click=delete_thread,
            args=(st.session_state.thread_index,),
        )
        # if st.button(label="$\;$Refresh the screen$~$"):
        #     st.rerun()

        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Aug. 2023  \n</small>",
            "<small>[ChatGPT (RAG) & DALL·E](https://chatgpt-dalle.streamlit.app/)  \n</small>",
            "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
            "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
            unsafe_allow_html=True,
        )
        # if st.button("Finish"):
        #     os._exit(0)

    if st.session_state.run_assistants:
        run_assistant(model, assistant_id, thread_index)
    else:
        manage_assistant(assistant_id)


if __name__ == "__main__":
    openai_assistants()
