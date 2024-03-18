"""
OpenAI Assistants using openai API (by T.-W. Yoon, Jan. 2024)
"""

import streamlit as st
import os
import time
import requests
import pickle
import hashlib
import json
from openai import OpenAI, APIError
from io import BytesIO
from PIL import Image
from tavily import TavilyClient
from audio_recorder_streamlit import audio_recorder


class NamedBytesIO(BytesIO):
    def __init__(self, buffer, name: str):
        super().__init__(buffer)
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()  # Close the buffer and free up the resources


def is_openai_api_key_valid(openai_api_key):
    """
    Return True if the given OpenAI api_key is valid.
    """

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
    }
    response = requests.get(
        "https://api.openai.com/v1/models", headers=headers
    )

    return response.status_code == 200


def tavily_search(query):
    """
    Perform a search using the Tavily API based on the provided query.
    """

    tavily_client = TavilyClient(api_key=st.session_state.tavily_api_key)
    search_result = tavily_client.get_search_context(
        query,
        search_depth="advanced",
        max_tokens=8192
    )

    return search_result


def submit_tool_outputs(thread_id, run_id, tools_to_call):
    """
    Submit tool outputs for a specific thread and run
    using the provided list of tools to call.
    """

    client = st.session_state.client
    tool_output_array = []

    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name == "tavily_search":
            output = tavily_search(query=json.loads(function_args)["query"])

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )


def run_and_wait_for_results(run, thread_id):
    """
    Take a run object and its thread id as input, and return True or False
    depending on whether the run is successfully completed or not.
    """

    while run.status != "completed":
        run = st.session_state.client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run.id
        )
        if run.status in {"failed", "expired"}:
            st.error(f"The API request has {run.status}.", icon="🚨")
            return False
        elif run.status == "requires_action":
            try:
                run = submit_tool_outputs(
                    thread_id,
                    run.id,
                    run.required_action.submit_tool_outputs.tool_calls
                )
            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")
                run = st.session_state.client.beta.threads.runs.cancel(
                    thread_id=thread_id, run_id=run.id
                )
                return False
        time.sleep(0.5)

    return True


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
        messages_list: The list of the most recent assistant message objects
                       in the conversation thread,
                       or None if the request has not been completed.
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
    except APIError as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        return None

    # Periodically retrieve the Run to check status and see if it has completed
    with st.spinner("AI is thinking..."):
        if run_and_wait_for_results(run, thread_id):
            messages = st.session_state.client.beta.threads.messages.list(
                thread_id=thread_id,
                order="desc"
            )

            messages_list = []
            index = 0
            while messages.data[index].role == "assistant":
                messages_list.insert(0, messages.data[index])
                index += 1

            return messages_list
        else:
            return None


def process_citations(content):
    """
    Process citations in the given message,
    and returns the modified message, citations, and cited files.

    Args:
        content: The content of a message object.

    Returns:
        tuple: A tuple containing the modified message content,
               citations (list), cited files (list) and annotation files (list).
    """
    client = st.session_state.client

    annotations = content.annotations
    citations, cited_files, annotation_files = [], [], []

    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        content.value = content.value.replace(annotation.text, f" [{index+1}]")

        # Gather citations based on annotation attributes
        try:
            if (file_citation := getattr(annotation, "file_citation", None)):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(file_citation.quote)
                cited_files.append(f"[{index+1}] :blue[{cited_file.filename}]")
            elif (file_path := getattr(annotation, "file_path", None)):
                annotation_files.append(client.files.retrieve(file_path.file_id))
                link = f"https://platform.openai.com/files/{file_path.file_id}"
                content.value = content.value.replace(
                    f"]( [{index+1}])", f"]({link})"
                )
        except Exception as e:
            # st.error(f"An error occurred: {e}", icon="🚨")
            # Ignore if there are problems with extracting citation information
            pass

    return content.value, citations, cited_files, annotation_files


def get_file_path(number, length=20):
    """
    Return a file path of a given length using the hashed value of number.
    """

    hashed = hashlib.sha256(str(number).encode("utf-8")).hexdigest()
    file_name = f"files/{hashed[:length]}.pickle"

    return file_name


def get_file_name_from_id(file_id):
    """
    Return the file name corresponding to the given file id.
    """

    try:
        file = st.session_state.client.files.retrieve(file_id)
        file_name= file.filename
    except APIError:
        file_name = "deleted file"
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


def display_text_with_equations(text):
    # Replace inline LaTeX equation delimiters \\( ... \\) with $
    modified_text = text.replace("\\(", "$").replace("\\)", "$")

    # Replace block LaTeX equation delimiters \\[ ... \\] with $$
    modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")

    # Use st.markdown to display the formatted text with equations
    st.markdown(modified_text)


def show_image(file_id):
    # Show the image of the given file id
    client = st.session_state.client
    resp = client.files.with_raw_response.retrieve_content(file_id)
    if resp.status_code == 200:
        image_data = BytesIO(resp.content)
        img = Image.open(image_data)
        st.image(img)


def add_file_id_to_list(file_id, file_id_list):
    # Add file id to the given list of file ids
    if file_id not in file_id_list:
        file_id_list.append(file_id)
        update_threads_info()


def show_messages(message_data_list):
    """
    Show the given list of messages.
    """

    thread_index = st.session_state.thread_index
    file_id_list = st.session_state.threads_list[thread_index]["file_ids"]

    for message in message_data_list:
        if message.file_ids:
            msg_files = []
            for file_id in message.file_ids:
                msg_files.append(
                    f"[{get_file_name_from_id(file_id)}]" +
                    f"(https://platform.openai.com/files/{file_id})"
                )
            msg_files = " (" + ", ".join(msg_files) + ")"
        else:
            msg_files = ""

        for message_content in message.content:
            if message.role == "user":
                with st.chat_message("user"):
                    # st.markdown(message_content.text.value + msg_files)
                    display_text_with_equations(message_content.text.value + msg_files)
            elif (text := getattr(message_content, "text", None)):
                # Extract the annotation information together with the text
                content_text, citations, cited_files, annotation_files = (
                    process_citations(text)
                )
                with st.chat_message("assistant"):
                    # st.markdown(content_text + msg_files)
                    display_text_with_equations(content_text)
                    if citations:
                        with st.expander("Source(s)"):
                            for citation, file in zip(citations, cited_files):
                                st.markdown(file, help=citation)
                    if annotation_files:
                        with st.expander("File(s) created by the assistant"):
                            for file in annotation_files:
                                link = f"https://platform.openai.com/files/{file.id}"
                                st.markdown(f"[{file.filename}]({link})")
                                add_file_id_to_list(file.id, file_id_list)
            elif (image_file := getattr(message_content, "image_file", None)):
                # Display the image generated by the assistant
                try:
                    file_id = image_file.file_id
                    show_image(file_id)
                    add_file_id_to_list(file_id, file_id_list)
                except Exception as e:
                    st.error(f"An error occurred: {e}", icon="🚨")


def show_thread_messages(thread_id, no_of_messages="All"):
    """
    Show the most recent 'no_of_messages' messages of a given thread.
    The argument 'no_of_messages' is a positive integer or "All, and
    if 'no_of_messages' is "All", all the messages are shown.
    """

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=thread_id,
        order="asc"
    )

    if no_of_messages == "All":
        no_of_messages = len(messages.data)
    elif not isinstance(no_of_messages, int) or no_of_messages <= 0:
        st.error("'no_of_messages' is a positive integer or 'All'", icon="🚨")
        return None

    show_messages(messages.data[-no_of_messages:])


def name_thread(thread_id):
    """
    Name the thread with the given ID using a summary of the first user query.
    """

    if not thread_exists(thread_id):
        return None

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=thread_id,
        order="asc"
    )

    first_query = messages.data[0].content[0].text.value

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


def get_thread(thread_id=None):
    """
    Get the thread with the given ID, and add it to the top of the threads list
    containing the thread ids, names and a list of file ids. If the given ID is
    None, a new thread is created. The name of this new thread is "No name yet",
    which will be replaced by a proper name when the thread is used.
    """

    if thread_id is None:
        thread = st.session_state.client.beta.threads.create()
        thread_id = thread.id
        thread_name = "No name yet"
    else:
        thread_name = name_thread(thread_id)

    st.session_state.threads_list.insert(
        0, {"id": thread_id, "name": thread_name, "file_ids": []}
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
            for assistant_file_id in assistant.file_ids:
                if assistant_file_id == file_id:
                    try:
                        # Delete the association with assistants
                        client.beta.assistants.files.delete(
                            assistant_id = assistant.id,
                            file_id=file_id
                        )
                    except APIError as e:
                        # st.error(f"An error occurred: {e}", icon="🚨")
                        pass

    try:
        client.files.delete(file_id)
    except APIError as e:
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
        get_thread(None)


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
        key="upload" + str(st.session_state.uploader_key),
    )

    if uploaded_files:
        uploaded_file_ids = []
        for file in uploaded_files:
            # Use BytesIO to read the file content
            # with BytesIO(file.getbuffer()) as in_memory:
            with NamedBytesIO(file.getbuffer(), file.name) as in_memory:
                try:
                    response = st.session_state.client.files.create(
                        file=in_memory,
                        purpose="assistants",
                    )
                    uploaded_file_ids.append(response.id)
                except APIError as e:
                    st.error(f"An error occurred: {e}", icon="🚨")
                    return []
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
        unsorted = [
            (assistant.name, assistant.id) for assistant in assistants
        ]
        st.session_state.assistants_name_id = sorted(
            unsorted, key=lambda x: (x[0] is None, x[0])
        )
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
        except APIError as e:
            # st.error(f"An error occurred: {e}", icon="🚨")
            pass
    try:
        st.session_state.client.beta.assistants.delete(assistant_id)
    except APIError as e:
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
        st.error(f"An error occurred: {e}", icon="🚨")

    return text


def input_from_mic():
    """
    Convert audio input from mic to text and returns it.
    If there is no audio input, None is returned.
    """

    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes == st.session_state.audio_bytes or audio_bytes is None:
        return None
    else:
        st.session_state.audio_bytes = audio_bytes
        return read_audio(audio_bytes)


def show_assistant(assistant_id):
    """
    Show the information of an assistant object
    """

    client = st.session_state.client
    assistant = client.beta.assistants.retrieve(assistant_id)
    tools = []
    for tool in assistant.tools:
        tool_name = tool.type if tool.type != "function" else tool.function.name
        tools.append(tool_name)
    files = [client.files.retrieve(file_id) for file_id in assistant.file_ids]
    file_ids, file_names = [], []

    # Show the file names and ids up to 12 characters
    for file in files:
        file_ids.append(file.id[0:12] + "...")
        file_name = file.filename
        if len(file_name) > 12:
            file_name = file_name[0:12] + "..."
        file_names.append(file_name)

    st.write("")
    if st.button(label="Create an assistant"):
        st.session_state.manage_assistant_app = "create"
        st.rerun()

    st.write("")
    st.write("**Assistant**")

    if assistant_id is not None:
        st.write(
            f"- :blue[Name]: {assistant.name}\n"
            f"- :blue[Default Model]: {assistant.model}\n"
            f"- :blue[ID]: {assistant.id}\n"
            f"- :blue[Instructions]: {assistant.instructions}\n"
            f"- :blue[Tool(s)]: {', '.join(tools)}\n"
            f"- :blue[File Name(s)]: {', '.join(file_names)}\n"
            f"- :blue[File ID(s)]: {', '.join(file_ids)}"
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

    query_description = (
        "The search query to use. For example: 'Latest news on Nvidia stock performance'"
    )
    tavily_search = {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Get information on recent events from the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": query_description},
                },
                "required": ["query"]
            }
        }
    }
    functions = {"tavily_search": tavily_search}
    available_tools = ["retrieval", "code_interpreter", "tavily_search"]

    model_options = ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]
    if assistant_id is None:
        st.write("**:blue[Create your assistant]**")
        model_index = 0
        assistant_name_value = ""
        instructions_value = ""
        tools_value = None
    else:
        assistant = st.session_state.client.beta.assistants.retrieve(
            assistant_id
        )
        st.write(f"**:blue[Modify the assistant] $\,${assistant.name}**")
        if assistant.model in model_options:
            model_index = model_options.index(assistant.model)
        else:
            model_index = 0
        assistant_name_value = assistant.name
        instructions_value = assistant.instructions
        tools_value = []
        for tool in assistant.tools:
            tool_name = tool.type if tool.type != "function" else tool.function.name
            tools_value.append(tool_name)
        file_ids_value = assistant.file_ids

    with st.form("Submit"):
        st.write("**Name** $\,$(Do not press Enter.)")
        name = st.text_input(
            label="assistant name",
            value=assistant_name_value,
            label_visibility="collapsed",
        )
        st.write(
            """
            **Model** $\,$(This default model will be overriden
            by the model selected at the time of running threads.)
            """
        )
        model = st.radio(
            label="Default models",
            options=("gpt-3.5-turbo-0125", "gpt-4-0125-preview"),
            label_visibility="collapsed",
            index=model_index,
        )
        st.write("**Instructions**")
        instructions = st.text_area(
            label="instructions",
            value=instructions_value,
            label_visibility="collapsed",
        )
        st.write(
            """
            **Tools** $\,$(:blue[tavily_search] is implemented using
            'function calling'.)
            """
        )
        tool_options = available_tools
        tool_names = st.multiselect(
            label="assistant tools",
            options=tool_options,
            default=tools_value,
            label_visibility="collapsed",
        )
        tools = []
        for tool_name in tool_names:
            if tool_name in ("retrieval", "code_interpreter"):
                tool = {"type": tool_name}
            else:
                tool = functions[tool_name]
            tools.append(tool)
        st.write("**File(s)**$\,$ (to be added)")
        file_ids = upload_files()
        if assistant_id:
            file_ids.extend(file_ids_value)

        form_left, form_right = st.columns(2)
        submitted = form_left.form_submit_button("Submit")
        if submitted:
            try:
                if assistant_id is None:
                    st.session_state.client.beta.assistants.create(
                        model=model,
                        name=name,
                        instructions=instructions,
                        tools=tools,
                        file_ids=file_ids,
                    )
                else:
                    st.session_state.client.beta.assistants.update(
                        assistant_id=assistant_id,
                        model=model,
                        name=name,
                        instructions=instructions,
                        tools=tools,
                        file_ids=file_ids,
                    )
                set_assistants_list()
                st.session_state.manage_assistant_app = "show"
                st.rerun()
            except APIError as e:
                st.error(f"An error occurred: {e}", icon="🚨")

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


def run_assistant(model, assistant_id):
    """
    Run a conversation with the specified assistant.

    Args:
        model (str): The model used for the assistant.
        assistant_id (str): The ID of the assistant.

    Returns:
        None
    """

    st.write("##### Conversation with Assistant")
    st.write("")

    thread_index = st.session_state.thread_index
    thread_id = st.session_state.threads_list[thread_index]["id"]

    if st.session_state.threads_list:
        show_thread_messages(thread_id, st.session_state.no_of_messages)

    query = st.chat_input(
        placeholder="Enter your query",
    )

    if query or st.session_state.text_from_audio:
        if st.session_state.text_from_audio:
            query = st.session_state.text_from_audio

        with st.chat_message("user"):
            st.markdown(query)

        # Append the file ids in this message to st.session_state.threads_list
        if st.session_state.file_ids:
            file_id_list = st.session_state.threads_list[thread_index]["file_ids"]
            for file_id in st.session_state.file_ids:
                add_file_id_to_list(file_id, file_id_list)

        message = run_thread(
            model, assistant_id, thread_id, query, st.session_state.file_ids
        )

        st.session_state.file_ids = []
        st.session_state.uploader_key += 1

        if message is None:
            st.error("Request not completed.", icon="🚨")
        else:
            show_messages(message)
            if st.session_state.threads_list[thread_index]["name"] == "No name yet":
                thread_name = name_thread(thread_id)
                st.session_state.threads_list[thread_index]["name"] = thread_name
                update_threads_info()

        if st.session_state.text_from_audio:
            st.session_state.text_from_audio = None
            st.rerun()

    # st.session_state.file_ids = upload_files(["pdf", "txt"])
    st.session_state.file_ids = upload_files()
    assistants_name_id = st.session_state.assistants_name_id
    assistant_index = st.session_state.assistant_index
    st.markdown(
        f"<small>Thread ID: :blue[{thread_id}] </small>"
        f"<small>(currently with :blue[{assistants_name_id[assistant_index][0]}] </small>"
        f"<small>and :blue[{model}])</small>",
        unsafe_allow_html=True
    )

    # Use your microphone
    st.session_state.text_from_audio = input_from_mic()
    if st.session_state.text_from_audio is not None:
        st.rerun()


def openai_assistants():
    """
    This main function presents OpenAI assistants by managing assistants,
    threads, messages, and files.
    """

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

    if "thread_id_input_key" not in st.session_state:
        st.session_state.thread_id_input_key = 0

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

    st.write("## 📚 OpenAI Assistants")
    st.write("")

    with st.sidebar:
        st.write("")
        st.write("**API Key Selection**")
        choice_api = st.sidebar.radio(
            label="$\\textsf{API Key Selection}$",
            options=("Your keys", "My keys"),
            label_visibility="collapsed",
            horizontal=True,
        )
        if choice_api == "Your keys":
            st.write("**OpenAI API Key**")
            st.session_state.openai_api_key = st.text_input(
                label="$\\textsf{Your OPenAI API Key}$",
                type="password",
                placeholder="sk-",
                label_visibility="collapsed",
            )
            st.write("**Tavily Search API Key**")
            st.session_state.tavily_api_key = st.text_input(
                label="$\\textsf{Your Tavily API Key}$",
                type="password",
                placeholder="tvly-",
                label_visibility="collapsed",
            )
            authentication = True
        else:
            st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state.tavily_api_key = st.secrets["TAVILY_API_KEY"]
            stored_pin = st.secrets["USER_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="$\\textsf{Password}$",
                type="password",
                label_visibility="collapsed",
            )
            authentication = user_pin == stored_pin

    if authentication:
        if is_openai_api_key_valid(st.session_state.openai_api_key):
            st.session_state.client = OpenAI(
                api_key=st.session_state.openai_api_key
            )
            # Set the variable st.session_state.assistants_name_id
            # containing assistant names and ids
            if not st.session_state.assistants_name_id:
                set_assistants_list()
        else:
            st.info(
                """
                **Enter your OpenAI and Tavily Search API keys in the sidebar**

                Get an OpenAI API key [here](https://platform.openai.com/api-keys)
                and a Tavily Search API key [here](https://app.tavily.com/). If you
                do not want to use any search tool, no need to enter your
                Tavily Search API key.
                """
            )
            st.info(
                """
                **Which information is stored where?**

                Objects like assistants, threads, and messages are all
                stored on OpenAI. In the Streamlit server where this app
                is being deployed, only lists containing the IDs and names
                of the thread objects are maintained. The problem is that
                the lists may be initialized when the app is rebooted.
                Users are therefore encouraged to save the thread IDs,
                as they can be used to recover unlisted (missing) threads.
                Thread IDs are shown at the bottom of each thread message.
                """
            )
            with st.expander("Sample Assistant Instructions"):
                st.markdown(sample_instructions)
            st.stop()
    else:
        st.info("**Enter the correct password in the sidebar**")
        st.stop()

    # Set the file name for storing thread information
    if "thread_info_pickle" not in st.session_state:
        st.session_state.thread_info_pickle = get_file_path(
            st.session_state.openai_api_key
        )

    if not os.path.exists(st.session_state.thread_info_pickle):
        # Create a thread and save the corresponing session state to pickle
        get_thread(None)

    # Load st.session_state.thread_dictionary from pickle
    load_thread_info_file()

    with st.sidebar:
        st.write("**Models**")
        model = st.radio(
            label="$\\textsf{Models}$",
            options=("gpt-3.5-turbo-0125", "gpt-4-0125-preview"),
            label_visibility="collapsed",
            index=0,
        )

        # Make a list of assistant names
        assistant_names = [
            name for (name, _) in st.session_state.assistants_name_id
        ]
        assistant_names = make_unique_names(assistant_names)
        st.write("**Assistant Names**")
        if assistant_names:
            if st.session_state.assistant_index >= len(assistant_names):
                st.session_state.assistant_index = 0
            assistant_name = st.selectbox(
                label="$\\textsf{Assistant Names}$",
                options=assistant_names,
                label_visibility="collapsed",
                index=st.session_state.assistant_index,
            )
            index = assistant_names.index(assistant_name)
            assistant_id = st.session_state.assistants_name_id[index][1]
            if index != st.session_state.assistant_index:
                st.session_state.assistant_index = index
                st.rerun()

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

        st.write("**Thread Names**")
        thread_name = st.selectbox(
            label="$\\textsf{Thread Names}$",
            options=st.session_state.thread_names,
            label_visibility="collapsed",
            index=st.session_state.thread_index
        )
        thread_index = st.session_state.thread_names.index(thread_name)
        if thread_index != st.session_state.thread_index:
            st.session_state.thread_index = thread_index
            st.rerun()

        thread_id = st.session_state.threads_list[thread_index]["id"]
        if thread_exists(thread_id):
            st.session_state.thread_index = thread_index
        else:
            delete_thread(thread_index)
            st.rerun()

        st.write(
            "**Thread ID** "
            "<small>(for Unlisted Thread)</small>", unsafe_allow_html=True
        )
        thread_id_input = st.text_input(
            label="$\\textsf{Thread ID}$",
            value="",
            label_visibility="collapsed",
            key="text_input" + str(st.session_state.thread_id_input_key)
        )
        if thread_id := thread_id_input.strip():
            if thread_exists(thread_id):
                found = False
                for index, id_name in enumerate(st.session_state.threads_list):
                    if thread_id == id_name["id"]:
                        st.session_state.thread_index = index
                        found = True
                        break
                if not found:
                    get_thread(thread_id)

            st.session_state.thread_id_input_key += 1
            st.rerun()

        st.write("**Prev. Messages to Show**")
        st.session_state.no_of_messages = st.radio(
            label="$\\textsf{Messages to show}$",
            options=("All", 20, 10),
            label_visibility="collapsed",
            horizontal=True,
            index=2,
        )

        st.write("")
        st.button(
            label="Create a new thread",
            on_click=get_thread,
            args=(None,)
        )
        st.button(
            label="$\:\,$Delete this thread$\:\,$",
            on_click=delete_thread,
            args=(st.session_state.thread_index,),
        )
        if st.button(label="$\;$Refresh the screen$~$"):
            st.rerun()

        st.write(
            "<small>Deleting a thread will also remove the files used in the thread.</small>",
            unsafe_allow_html=True,
        )

        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Jan. 2024  \n</small>",
            "<small>[ChatGPT (RAG) & DALL·E](https://chatgpt-dalle.streamlit.app/)  \n</small>",
            "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
            "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
            unsafe_allow_html=True
        )

    if st.session_state.run_assistants:
        run_assistant(model, assistant_id)
    else:
        manage_assistant(assistant_id)


# Sample instructions
sample_instructions = """
:blue[Search Only]:

You are a helpful assistant. Your goal is to provide answers to human
inquiries using search results from the internet through the 'tavily_search'
function. Your answers should be solely based on information from the internet,
not from your general knowledge. Use markdown syntax and include relevant URL
sources following MLA format. Should the information not be available through
the 'tavily_search' function, please inform the human explicitly that the
answer could not be found.

:blue[Search]:

You are a helpful assistant. Your goal is to provide answers to human
inquiries using 1) search results from the internet through the
'tavily_search' function or 2) your general knowledge. You must inform
the human of the basis of your answers, i.e., whether your answers are
based on 1) or 2). Use markdown syntax and include relevant URL sources
following MLA format. Should the information not be available through
the 'tavily_search' function or your general knowledge, please inform
the human explicitly that the answer could not be found.

:blue[Retrieval Only]:

You are a helpful assistant. Your goal is to provide answers to human
inquiries using information from the uploaded document. Your answers
should be solely based on information from the uploaded documents,
not from your general knowledge. Use markdown syntax and include relevant
sources following MLA format. Should the information not be available
through the uploaded documents, please inform the human explicitly that
the answer could not be found.

:blue[Retrieval]:

You are a helpful assistant. Your goal is to provide answers to human
inquiries using 1) information from the uploaded documents or 2) your
general knowledge. You must inform the human of the basis of your
answers, i.e., whether your answers are based on 1) or 2). Use markdown
syntax and include relevant sources following MLA format. Should
the information not be available through the uploaded documents or
your general knowledge, please inform the human explicitly that
the answer could not be found.

:blue[Retrieval & Search]:

You are a helpful assistant. Your goal is to provide answers to human
inquiries using 1) information from the uploaded documents, 2) search
results from the internet through the 'tavily_search' function,
or 3) your general knowledge. You must inform the human of the basis
of your answers, i.e., whether your answers are based on 1), 2), or 3).
Use markdown syntax and include relevant sources, like URLs, following
MLA format. Should the information not be available through the
uploaded documents, the 'tavily_search' function, or your general
knowledge, please inform the human explicitly that the answer could
not be found.

:blue[Python Assistant]:

You specialize in Python programming as a coding assistant. Your task
is to write Python code that fulfills the user's requirements.
Additionally, if given the user's code, carefully analyze it to:

1. Identify any errors or bugs.
2. Suggest ways for optimizing code efficiency and structure.
3. Recommend enhancements to improve code readability and maintainability.

Run code whenever necessary to verify its functionality and to identify
potential improvements. Your feedback should aim to help the user enhance
their coding skills and adopt best coding practices. If your response
includes information obtained through the 'tavily_search' function,
please include relevant URL sources following MLA format. Your primary
goal is to assist users in becoming more proficient and efficient Python
developers.
"""


if __name__ == "__main__":
    openai_assistants()
