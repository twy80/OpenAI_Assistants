"""
OpenAI Agent using the Assistants API (by T.-W. Yoon, May 2024)
"""

import streamlit as st
import os
import time
import requests
import pickle
import hashlib
import json
from openai import OpenAI, APIError, AssistantEventHandler
from io import BytesIO
from PIL import Image
from langchain_community.utilities import BingSearchAPIWrapper
from audio_recorder_streamlit import audio_recorder
from typing_extensions import override
# The following are for type annotations
from typing import Union, List, Tuple, Dict, Literal, Optional
from streamlit.runtime.uploaded_file_manager import UploadedFile
from openai.types.beta.threads.message import Message
from openai.types.beta.threads.text import Text

GPT3_5, GPT4 = "gpt-3.5-turbo", "gpt-4o"


class NamedBytesIO(BytesIO):
    def __init__(self, buffer, name: str) -> None:
        super().__init__(buffer)
        self.name = name

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()  # Close the buffer and free up the resources


def is_openai_api_key_valid(openai_api_key: str) -> bool:
    """
    Return True if the given OpenAI API key is valid.
    """

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
    }
    response = requests.get(
        "https://api.openai.com/v1/models", headers=headers
    )

    return response.status_code == 200


def is_bing_subscription_key_valid(bing_subscription_key: str) -> bool:
    """
    Return True if the given Bing subscription key is valid.
    """

    if not bing_subscription_key:
        return False
    try:
        search = BingSearchAPIWrapper(
            bing_subscription_key=bing_subscription_key,
            bing_search_url="https://api.bing.microsoft.com/v7.0/search",
            k=1
        )
        search.run("Where can I get a Bing subscription key?")
    except:
        return False
    else:
        return True


def check_api_keys() -> None:
    # Unset this flag to check the validity of the OpenAI API key
    st.session_state.ready = False


def bing_search(query: str) -> str:
    """
    Search the internet for the provided query
    using the Bing Subscription Key.
    """

    search = BingSearchAPIWrapper(
        bing_subscription_key=st.session_state.bing_subscription_key,
        bing_search_url="https://api.bing.microsoft.com/v7.0/search",
    )
    list_of_results = search.results(query=query, num_results=5)

    results = ""
    for entry in list_of_results:
        results += f"snippet: {entry['snippet']}\n"
        results += f"title: {entry['title']}\n"
        results += f"url: {entry['link']}\n\n"

    return results


class EventHandler(AssistantEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_placeholder = st.empty()
        self.current_text = ""

    @override
    def on_text_created(self, text):
        # self.current_text = "\nassistant > "
        self.text_placeholder.write(self.current_text)

    @override
    def on_text_delta(self, delta, snapshot):
        self.current_text += delta.value
        self.text_placeholder.write(self.current_text)

    def on_tool_call_created(self, tool_call):
        if tool_call.type == "function":
            tool_name = "Search"
        else:
            tool_name = tool_call.type
            tool_name = tool_name[0].upper() + tool_name[1:]
        self.current_text += f"\n\n**:blue[{tool_name}]**: "
        self.text_placeholder.write(self.current_text)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                self.current_text += delta.code_interpreter.input
                self.text_placeholder.write(self.current_text)
            if delta.code_interpreter.outputs:
                self.current_text += f"\n\n**:blue[Output]**: "
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        self.current_text += f"\n\n{output.logs}"
                self.text_placeholder.write(self.current_text)
                self.current_text += "\n\n**:blue[Assistant]**: "

    @override
    def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == 'thread.run.requires_action':
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.handle_requires_action(event.data, run_id)

    def handle_requires_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            output = None
            tool_call_id = tool.id
            function_name = tool.function.name
            function_args = tool.function.arguments

            if function_name == "bing_search":
                output = bing_search(query=json.loads(function_args)["query"])

            if output:
                tool_outputs.append({"tool_call_id": tool_call_id, "output": output})

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        # Use the submit_tool_outputs_stream helper
        with st.session_state.client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()


def create_message_run_stream(
    thread_id: str,
    query: str,
    attached_files: Dict[str, str]
) -> Optional[Message]:

    """
    Run a conversation thread with an assistant.

    Args:
        thread_id: The ID of the conversation thread.
        query: The user's query.
        attached_files: Dictionary containing list of attached file IDs.
    Returns:
        Message object
    """

    content = [{"type": "text", "text": query}]
    image_files = attached_files["image"]
    code_interpreter_files = attached_files["code_interpreter"]
    file_search_files = attached_files["file_search"]

    if image_files:
        img_files = [
            {"type": "image_file", "image_file": {"file_id": file_id}}
            for file_id in image_files
        ]
        content.extend(img_files)

    code_interpreter_attachments = [
        {"file_id": file_id, "tools": [{"type": "code_interpreter"}]}
        for file_id in code_interpreter_files
    ]
    file_search_attachments = [
        {"file_id": file_id, "tools": [{"type": "file_search"}]}
        for file_id in file_search_files
    ]

    try:
        # Create the user message and add it to the thread
        message = st.session_state.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
            attachments=code_interpreter_attachments + file_search_attachments,
        )
        return message

    except APIError as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        return None


def show_most_recent_assistant_image(thread_id: str) -> None:
    """
    Show the most recent assistant image.
    """

    messages = st.session_state.client.beta.threads.messages.list(
        thread_id=thread_id,
        order="desc"
    )

    for message in messages.data:
        if message.role == "user":
            break
        for message_content in message.content:
            if (image_file := getattr(message_content, "image_file", None)):
                show_image(image_file.file_id)


def process_citations(
    content: Text
) -> Tuple[str, List[Optional[str]], List[str], List[str]]:

    """
    Process citations in the given message, and return
    the modified message, citations, cited files, and cited links.

    Args:
        content: The content of a message object.

    Returns:
        A tuple containing the modified message content,
        and lists of citations, cited files, and cited links.
    """

    client = st.session_state.client

    annotations = content.annotations
    citations, cited_files, cited_links = [], [], []

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
                cited_file = client.files.retrieve(file_path.file_id)
                link = f"https://platform.openai.com/storage/files/{file_path.file_id}"
                cited_links.append(
                    f"[{index+1}] [:blue[{cited_file.filename}]]({link})"
                )
        except Exception as e:
            # st.error(f"An error occurred: {e}", icon="🚨")
            # Ignore if there are problems with extracting citation information
            pass

    return content.value, citations, cited_files, cited_links


def get_file_path(key: str, length: int=20) -> str:
    """
    Return a file path of a given length using the hashed value of the key.
    """

    hashed = hashlib.sha256(key.encode("utf-8")).hexdigest()
    file_name = f"files/{hashed[:length]}.pickle"

    return file_name


def get_file_name_from_id(file_id: str) -> str:
    """
    Return the file name corresponding to the given file id.
    """

    try:
        file = st.session_state.client.files.retrieve(file_id)
        file_name = file.filename
    except APIError:
        file_name = "deleted file"
    return file_name


def get_file_names_ids(file_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Take a list of file IDs as input and return a list of the corresponding
    file names. The first 12 characters of the file IDs are also returned.
    """

    file_id_heads, file_names = [], []

    # Show the file names and ids up to 12 characters
    for file_id in file_ids:
        file_name = get_file_name_from_id(file_id)
        if len(file_name) > 12:
            file_name = file_name[0:12] + "..."
        file_names.append(file_name)
        file_id_heads.append(file_id[0:12] + "...")

    return file_names, file_id_heads


def get_vector_store_name_from_id(vector_store_id: str) -> str:
    """
    Return the vector store name corresponding to the given file id.
    """

    client = st.session_state.client
    try:
        vector_store = client.beta.vector_stores.retrieve(vector_store_id)
        vector_store_name = vector_store.name
    except APIError:
        vector_store_name = "deleted vector store"
    return vector_store_name


def thread_exists(thread_id: str) -> bool:
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


def display_text_with_equations(text: str) -> None:
    """
    Modify text with equations for better viewing in markdown.
    """

    # Replace inline LaTeX equation delimiters \\( ... \\) with $
    modified_text = text.replace("\\(", "$").replace("\\)", "$")

    # Replace block LaTeX equation delimiters \\[ ... \\] with $$
    modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")

    # Use st.markdown to display the formatted text with equations
    st.markdown(modified_text)


def show_image(image_file: str) -> None:
    """
    Show the image of the given file id or url.
    """

    client = st.session_state.client
    byte_image = None
    try:
        if image_file.startswith("file-"):
            byte_image = client.files.content(image_file).read()
        elif image_file.startswith("http"):
            resp = requests.get(image_file)
            if resp.status_code == 200:
                byte_image = resp.content

        if byte_image is not None:
            img = Image.open(BytesIO(byte_image))
            st.image(img)
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")


def show_messages(messages: List[Message]) -> None:
    """
    Show the given list of messages.
    """

    thread_index = st.session_state.thread_index

    for message in messages:
        for message_content in message.content:
            if (text := getattr(message_content, "text", None)):
                content_text, citations, cited_files, cited_links = (
                    process_citations(text)
                )
                with st.chat_message(message.role):
                    display_text_with_equations(content_text)
                    if citations:
                        with st.expander("Source(s)"):
                            for citation, file in zip(citations, cited_files):
                                st.markdown(file, help=citation)
                    if cited_links:
                        with st.expander("File(s) created by the assistant"):
                            for cited_link in cited_links:
                                st.markdown(cited_link)
            elif (image_file := getattr(message_content, "image_file", None)):
                file_id = image_file.file_id
                file_name = get_file_name_from_id(file_id)
                st.session_state.threads_list[thread_index]["file_ids"].append(
                    file_id
                )
                if message.role == "assistant":
                    show_image(file_id)
                else:
                    link = f"https://platform.openai.com/storage/files/{file_id}"
                    st.write(f"$~~~~$Attachment: [{file_name}]({link})")
            elif (image_url := getattr(message_content, "image_url", None)):
                show_image(image_url.url)
        if message.attachments:
            file_names_list = []
            for attachment in message.attachments:
                file_id = attachment.file_id
                st.session_state.threads_list[thread_index]["file_ids"].append(
                    file_id
                )
                link = f"https://platform.openai.com/storage/files/{file_id}"
                file_names_list.append(
                    f"[{get_file_name_from_id(file_id)}]({link})"
                )
            file_names = ", ".join(file_names_list)
            st.write(f"$~~~~$Attachment(s): {file_names}")


def show_thread_messages(
    thread_id: str,
    no_of_messages: Union[Literal["All"], int]
) -> None:

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


def name_thread(thread_id: str) -> None:
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
            model=GPT3_5,
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


def save_thread_info_file() -> None:
    """
    Save a list containing the thread ids, names and a list of file ids
    used in the thread to a pickle file.
    """

    with open(st.session_state.thread_info_pickle, "wb") as file:
        pickle.dump(st.session_state.threads_list, file)
    time.sleep(0.5)


def load_thread_info_file() -> None:
    """
    Load a list containing the thread ids, names and a list of file ids
    from a pickle file, and reset the list of thread names.
    """

    with open(st.session_state.thread_info_pickle, "rb") as file:
        st.session_state.threads_list = pickle.load(file)
    st.session_state.thread_names = [
        thread["name"] for thread in st.session_state.threads_list
    ]


def update_threads_info() -> None:
    """
    Reset the list of thread names, and save a list containing the thread
    ids, names and a list of file ids to a pickle file.
    """

    st.session_state.thread_names = [
        thread["name"] for thread in st.session_state.threads_list
    ]
    save_thread_info_file()


def get_thread(thread_id: Optional[str]) -> None:
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


def delete_file(file_id: str) -> None:
    """
    Delete the file of the given id.
    """

    try:
        st.session_state.client.files.delete(file_id)
    except APIError as e:
        # st.error(f"An error occurred: {e}", icon="🚨")
        pass


def delete_vector_store(vector_store_id: str) -> None:
    """
    Delete the vector store of the given id.
    """

    client = st.session_state.client

    vector_store_files = client.beta.vector_stores.files.list(
        vector_store_id=vector_store_id
    )
    file_ids = [file.id for file in vector_store_files.data]

    for file_id in file_ids:
        delete_file(file_id)

    try:
        client.beta.vector_stores.delete(vector_store_id)
    except APIError as e:
        # st.error(f"An error occurred: {e}", icon="🚨")
        pass


def delete_thread(thread_index: int) -> None:
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


def upload_files(
    purpose: Literal["file_search", "code_interpreter", "image"]
) -> List[UploadedFile]:

    """
    Upload files and return the list of the uploaded file ids.
    If no files are uploaded, return an empty list.
    """

    file_search = [
        "c", "cs", "cpp", "doc", "docx", "html", "java", "json", "md", "pdf",
        "php", "pptx", "py", "rb", "tex", "txt", "css", "js", "sh", "ts"
    ]
    image = ["jpeg", "jpg", "gif", "png"]
    supported_file_types = {
        "file_search": file_search,
        "image": image,
        "code_interpreter": (
            file_search + image + ["csv", "tar", "xlsx", "xml", "zip"]
        ),
    }
    type = supported_file_types.get(purpose)

    uploaded_files = st.file_uploader(
        label="Upload files",
        type=type,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=purpose + str(st.session_state.uploader_key),
    )

    return uploaded_files


def send_files_to_openai(
    uploaded_files: List[UploadedFile]
) -> List[str]:

    """
    Send a list of files to the OpenAI server and return
    the corresponding file ids.
    """

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


def make_unique_names(list_of_names: List[str]) -> List[str]:
    """
    Make all elments in list_of_names unique
    """

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


def show_files() -> None:
    st.write("")
    st.write("**File(s)** $\,$(Uploaded to OpenAI)")

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
        delete_file_button = st.button("Delete the file")
        if delete_file_button:
            st.warning("Are you sure you want to proceed?")
            left, right = st.columns(2)
            left.button(
                label="Yes, I'm sure.",
                on_click=delete_file,
                args=(file.id,),
                key="delete_file",
            )
            if right.button("No, I'm not"):
                st.rerun()
    else:
        st.write(":blue[No uploaded file]")


def show_vector_stores() -> None:
    st.write("")
    st.write("**Vector Store(s)**")

    vector_stores = st.session_state.client.beta.vector_stores.list().data
    if vector_stores:
        vector_store_names = [
            store.name if store.name else "No Name" for store in vector_stores
        ]
        vector_store_names = make_unique_names(vector_store_names)
        vector_store_name = st.selectbox(
            label="Vector store names",
            options=vector_store_names,
            label_visibility="collapsed",
            index=0,
        )
        index = vector_store_names.index(vector_store_name)
        vector_store = vector_stores[index]

        store_files = st.session_state.client.beta.vector_stores.files.list(
            vector_store_id=vector_store.id
        )
        file_ids = [file.id for file in store_files.data]
        store_file_names, _ = get_file_names_ids(file_ids)

        vector_store_name = vector_store.name if vector_store.name else "No Name"
        st.write(
            f"""
            - :blue[Vector Store Name]: {vector_store_name}
            - :blue[Vector Store ID]: {vector_store.id}
            - :blue[Vector Store File(s)]: {", ".join(store_file_names)}
            """
        )
        delete_vector_store_button = st.button("Delete the vector store")
        if delete_vector_store_button:
            st.warning("Are you sure you want to proceed?")
            left, right = st.columns(2)
            left.button(
                label="Yes, I'm sure.",
                on_click=delete_vector_store,
                args=(vector_store.id,),
                key="delete_vector_store",
            )
            if right.button("No, I'm not"):
                st.rerun()
    else:
        st.write(":blue[No vector store]")


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


def delete_assistant(assistant_id: str) -> None:
    """
    Delete the assistant of the given id along with
    the associated vector store.
    """

    assistant = st.session_state.client.beta.assistants.retrieve(assistant_id)

    if assistant.tool_resources.file_search is not None:
        vector_store_ids = assistant.tool_resources.file_search.vector_store_ids
        # Delete the vector store associated with the assistant
        for vector_store_id in vector_store_ids:
            delete_vector_store(vector_store_id)
    if assistant.tool_resources.code_interpreter is not None:
        # Delete the code interpreter files associated with the assistant
        ci_file_ids = assistant.tool_resources.code_interpreter.file_ids
        for file_id in ci_file_ids:
            delete_file(file_id)

    try:
        st.session_state.client.beta.assistants.delete(assistant_id)
        set_assistants_list()
    except APIError as e:
        # st.error(f"An error occurred: {e}", icon="🚨")
        pass


def run_or_manage_assistants() -> None:
    # Toggle the flag to determine whether to run or manage assistants
    st.session_state.run_assistants = not st.session_state.run_assistants


def read_audio(audio_bytes: bytes) -> Optional[str]:
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


def input_from_mic() -> Optional[str]:
    """
    Convert audio input from mic to text and returns it.
    If there is no audio input, None is returned.
    """

    time.sleep(0.5)
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes == st.session_state.audio_bytes or audio_bytes is None:
        return None
    else:
        st.session_state.audio_bytes = audio_bytes
        return read_audio(audio_bytes)


def show_assistant(assistant_id: str) -> None:
    """
    Show the information of an assistant object
    """

    client = st.session_state.client
    assistant = client.beta.assistants.retrieve(assistant_id)
    tools, ci_file_names, vector_store_id = [], [], ""

    for tool in assistant.tools:
        if tool.type == "function":
            tool_name = tool.function.name
        else:
            tool_name = tool.type
            if tool_name == "file_search":
                if assistant.tool_resources.file_search.vector_store_ids:
                    vector_store_id = (
                        assistant.tool_resources.file_search.vector_store_ids[0]
                    )
                else:
                    vector_store_id = ""
            elif tool_name == "code_interpreter":
                ci_file_names, _ = get_file_names_ids(
                    assistant.tool_resources.code_interpreter.file_ids
                )
        tools.append(tool_name)

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
            f"- :blue[Temperature]: {assistant.temperature}\n"
            f"- :blue[Tool(s)]: {', '.join(tools)}\n"
            f"- :blue[Vector Store ID]: {vector_store_id}\n"
            f"- :blue[Code Interpreter File(s)]: {', '.join(ci_file_names)}"
        )
        st.write("")
        left, right = st.columns(2)
        if left.button(label="Modify the assistant"):
            st.session_state.manage_assistant_app = "modify"
            st.rerun()
        if right.button("Delete the assistant"):
            st.warning("Are you sure you want to proceed?")
            left, right = st.columns(2)
            left.button(
                label="Yes, I'm sure.",
                on_click=delete_assistant,
                args=(assistant_id,),
                key="delete_assistant",
            )
            if right.button("No, I'm not"):
                st.rerun()
    else:
        st.write(":blue[No assistant yet]")

    show_vector_stores()
    show_files()


def add_files_to_vector_store(
    vector_store_file_ids: List[str], vector_store_id: str
) -> None:

    """
    Add the list of files to the vector store of the given ID.
    """

    try:
        st.session_state.client.beta.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store_id,
            file_ids=vector_store_file_ids,
        )
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")


def update_assistant(assistant_id: Optional[str]) -> None:
    """
    Update the assistant with 'assistant_id', or
    create an assistant when 'assistant_id' is None
    """

    query_description = (
        "The search query to use. For example: 'Latest news on Nvidia stock performance'"
    )
    bing_search = {
        "type": "function",
        "function": {
            "name": "bing_search",
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
    functions = {"bing_search": bing_search}
    if st.session_state.bing_subscription_validity:
        available_tools = ["file_search", "code_interpreter", "bing_search"]
    else:
        available_tools = ["file_search", "code_interpreter"]

    client = st.session_state.client
    model_options = [GPT3_5, GPT4]
    if assistant_id is None:
        st.write("**:blue[Create Your Assistant]**")
        model_index = 1
        assistant_name_value = ""
        instructions_value = ""
        tools_value = None
    else:
        assistant = client.beta.assistants.retrieve(
            assistant_id
        )
        st.write(f"**Modify $\,$:blue[{assistant.name}]**")
        if assistant.model in model_options:
            model_index = model_options.index(assistant.model)
        else:
            model_index = 1
        assistant_name_value = assistant.name
        instructions_value = assistant.instructions
        tools_value = []
        for tool in assistant.tools:
            if tool.type == "function":
                tool_name = tool.function.name
            else:
                tool_name = tool.type
            tools_value.append(tool_name)

    # with st.form("Submit"):
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
        options=(GPT3_5, GPT4),
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
        **Tools** $\,$(:blue[bing_search] is available
        if your bing subscription key is provided.)
        """
    )
    tool_names = st.multiselect(
        label="assistant tools",
        options=available_tools,
        default=tools_value,
        label_visibility="collapsed",
    )
    tools = []
    tool_resources = {}

    for tool_name in tool_names:
        if tool_name == "file_search":
            tool = {"type": tool_name}
            if (
                assistant_id is None or
                assistant.tool_resources.file_search is None
            ):
                st.session_state.vector_store_ids = []
            else:
                st.session_state.vector_store_ids = (
                    assistant.tool_resources.file_search.vector_store_ids
                )
            st.write("**:blue[Add vector store files]**")
            st.session_state.vector_store_files = upload_files(purpose="file_search")
        elif tool_name == "code_interpreter":
            tool = {"type": tool_name}
            if (
                assistant_id is None or
                assistant.tool_resources.code_interpreter is None
            ):
                st.session_state.ci_file_ids = []
            else:
                st.session_state.ci_file_ids = (
                    assistant.tool_resources.code_interpreter.file_ids
                )
            st.write("**:blue[Add code interpreter files]**")
            st.session_state.new_ci_files = upload_files(purpose="code_interpreter")
        else:
            tool = functions[tool_name]
        tools.append(tool)

    left, right = st.columns(2)
    submitted = left.button("Submit")
    if submitted:
        st.session_state.uploader_key += 1
        if st.session_state.vector_store_files:
            store_file_ids = send_files_to_openai(st.session_state.vector_store_files)
            if st.session_state.vector_store_ids:
                vector_store_id = st.session_state.vector_store_ids[0]
            else:
                vector_store = client.beta.vector_stores.create(
                    name=name
                )
                vector_store_id = vector_store.id
            add_files_to_vector_store(store_file_ids, vector_store_id)
            st.session_state.vector_store_ids = [vector_store_id]
        tool_resources["file_search"] = {
            "vector_store_ids": st.session_state.vector_store_ids
        }
        if st.session_state.new_ci_files:
            new_ci_file_ids = send_files_to_openai(st.session_state.new_ci_files)
            st.session_state.ci_file_ids.extend(new_ci_file_ids)
        tool_resources["code_interpreter"] = {
            "file_ids": st.session_state.ci_file_ids
        }

        try:
            if assistant_id is None:
                client.beta.assistants.create(
                    model=model,
                    name=name,
                    instructions=instructions,
                    tools=tools,
                    tool_resources=tool_resources,
                    temperature=st.session_state.temperature,
                )
            else:
                client.beta.assistants.update(
                    assistant_id=assistant_id,
                    model=model,
                    name=name,
                    instructions=instructions,
                    tools=tools,
                    tool_resources=tool_resources,
                    temperature=st.session_state.temperature,
                )
            set_assistants_list()
            st.session_state.manage_assistant_app = "show"
            st.rerun()
        except APIError as e:
            st.error(f"An error occurred: {e}", icon="🚨")

    back_to_manage = right.button("Back")
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
    file_options = "image", "code_interpreter", "file_search"

    if query or st.session_state.text_from_audio:
        if st.session_state.text_from_audio:
            query = st.session_state.text_from_audio

        with st.chat_message("user"):
            st.markdown(query)

        attached_files = {}
        for purpose in file_options:
            attached_files[purpose] = send_files_to_openai(
                st.session_state.files[purpose]
            )

        create_message_run_stream(
            thread_id=thread_id,
            query=query,
            attached_files=attached_files,
        )

        with st.chat_message("assistant"):
            try:
                with st.session_state.client.beta.threads.runs.stream(
                    model=model,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    temperature=st.session_state.temperature,
                    event_handler=EventHandler()
                ) as stream:
                    stream.until_done()
                    show_most_recent_assistant_image(thread_id)
                    st.session_state.files = {key: [] for key in file_options}
                    st.session_state.uploader_key += 1
                    if st.session_state.threads_list[thread_index]["name"] == (
                        "No name yet"
                    ):
                        thread_name = name_thread(thread_id)
                        st.session_state.threads_list[thread_index]["name"] = (
                            thread_name
                        )
                        update_threads_info()
            except APIError as e:
                st.error(f"An error occurred: {e}", icon="🚨")

        if st.session_state.text_from_audio:
            st.session_state.text_from_audio = None
            st.rerun()

    left, right = st.columns([1, 5])
    left.write("**$\:\!$Upload Files**")
    purpose = right.radio(
        label="purpose",
        options=file_options,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state.files[purpose] = upload_files(purpose)

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

    if "ready" not in st.session_state:
        st.session_state.ready = False

    if "bing_subscription_validity" not in st.session_state:
        st.session_state.bing_subscription_validity = False

    if "thread_index" not in st.session_state:
        st.session_state.thread_index = 0

    if "threads_list" not in st.session_state:
        st.session_state.threads_list = []

    if "thread_names" not in st.session_state:
        st.session_state.thread_names = []

    if "no_of_messages" not in st.session_state:
        st.session_state.no_of_messages = "All"

    if "files" not in st.session_state:
        st.session_state.files = {
            "image": [],
            "code_interpreter": [],
            "file_search": [],
        }

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

    if "manage_assistant_app" not in st.session_state:
        st.session_state.manage_assistant_app = "show"

    if "vector_store_ids" not in st.session_state:
        st.session_state.vector_store_ids = []

    if "vector_store_files" not in st.session_state:
        st.session_state.vector_store_files = []

    if "ci_file_ids" not in st.session_state:
        st.session_state.ci_file_ids = []

    if "new_ci_files" not in st.session_state:
        st.session_state.new_ci_files = []

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
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            st.write("**Bing Subscription Key**")
            st.session_state.bing_subscription_key = st.text_input(
                label="$\\textsf{Your Bing Subscription Key}$",
                type="password",
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            if is_bing_subscription_key_valid(
                st.session_state.bing_subscription_key
            ):
                st.session_state.bing_subscription_validity = True
            authentication = True
        else:
            st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state.bing_subscription_key = st.secrets["BING_SUBSCRIPTION_KEY"]
            st.session_state.bing_subscription_validity = True
            stored_pin = st.secrets["USER_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="$\\textsf{Password}$",
                type="password",
                label_visibility="collapsed",
            )
            authentication = user_pin == stored_pin

    if authentication:
        if not st.session_state.ready:
            if is_openai_api_key_valid(st.session_state.openai_api_key):
                st.session_state.client = OpenAI(
                    api_key=st.session_state.openai_api_key
                )
                st.session_state.ready = True
                # Set the variable st.session_state.assistants_name_id
                # containing assistant names and ids
                if not st.session_state.assistants_name_id:
                    set_assistants_list()
            else:
                st.info(
                    """
                    **Enter your OpenAI and Bing Subscription Keys in the sidebar**

                    Get an OpenAI API Key [here](https://platform.openai.com/api-keys)
                    and a Bing Subscription Key [here](https://portal.azure.com/).
                    If you do not want to use Bing Search for searching the internet,
                    no need to enter your Bing Subscription Key.
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
                st.image("files/Streamlit_Assistants_App.png")
                with st.expander("Sample Assistant Instructions"):
                    st.markdown(sample_instructions)
                st.info(
                    """
                    This app is coded by T.-W. Yoon, a professor of systems theory at
                    Korea University. Take a look at some of his other projects:
                    - [LangChain OpenAI Agent](https://langchain-openai-agent.streamlit.app/)
                    - [Multi-Agent Debate](https://multi-agent-debate.streamlit.app/)
                    - [TWY's Playground](https://twy-playground.streamlit.app/)
                    - [Differential equations](https://diff-eqn.streamlit.app/)
                    """
                )
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
            options=(GPT3_5, GPT4),
            label_visibility="collapsed",
            index=1,
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

        st.write("")
        st.write("**Temperature**")
        st.session_state.temperature = st.slider(
            label="Temperature (higher $\Rightarrow$ more random)",
            min_value=0.0,
            max_value=2.0,
            value=0.8,
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
        )

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
        delete_thread_button = st.button("$\:\,$Delete this thread$\:\,$")
        if delete_thread_button:
            st.warning("Are you sure you want to proceed?")
            left, right = st.columns(2)
            left.button(
                label="Yes.",
                on_click=delete_thread,
                args=(st.session_state.thread_index,),
                key="delete_thread",
            )
            if right.button("No"):
                st.rerun()
        if st.button(label="$\;$Refresh the screen$~$"):
            st.rerun()

        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Jan. 2024  \n</small>",
            "<small>[LangChain LLM Agent](https://langchain-llm-agent.streamlit.app/)  \n</small>",
            "<small>[Multi-Agent Debate](https://multi-agent-debate.streamlit.app/)  \n</small>",
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
inquiries using search results from the internet through the 'bing_search'
function. Your answers should be solely based on information from the internet,
not from your general knowledge. Use markdown syntax and include relevant URL
sources following MLA format. Should the information not be available through
the 'bing_search' function, please inform the human explicitly that the
answer could not be found.

:blue[Search]:

You are a helpful assistant. Your goal is to provide answers to human
inquiries using 1) search results from the internet through the
'bing_search' function or 2) your general knowledge. You must inform
the human of the basis of your answers, i.e., whether your answers are
based on 1) or 2). Use markdown syntax and include relevant URL sources
following MLA format. Should the information not be available through
the 'bing_search' function or your general knowledge, please inform
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
results from the internet through the 'bing_search' function,
or 3) your general knowledge. You must inform the human of the basis
of your answers, i.e., whether your answers are based on 1), 2), or 3).
Use markdown syntax and include relevant sources, like URLs, following
MLA format. Should the information not be available through the
uploaded documents, the 'bing_search' function, or your general
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
includes information obtained through the 'bing_search' function,
please include relevant URL sources following MLA format. Your primary
goal is to assist users in becoming more proficient and efficient Python
developers.
"""


if __name__ == "__main__":
    openai_assistants()
