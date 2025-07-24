import os
import validators
import traceback
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain.schema import Document
from urllib.parse import urlparse, parse_qs

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Main input
generic_url = st.text_input("URL", label_visibility="collapsed")

# Prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


# Custom YouTube transcript loader
def get_youtube_transcript_docs(video_url):
    video_id = parse_qs(urlparse(video_url).query).get("v")
    if not video_id:
        raise ValueError("Invalid YouTube URL.")
    video_id = video_id[0]

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript_list])
        return [Document(page_content=full_text)]
    except TranscriptsDisabled:
        raise Exception("❌ Subtitles are disabled for this video.")
    except NoTranscriptFound:
        raise Exception("❌ No transcript available for this video.")
    except Exception as e:
        raise Exception(f"❌ Failed to get transcript: {str(e)}")


# On button click
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video or website URL")
    else:
        try:
            with st.spinner("Loading content..."):
                # Initialize LLM
                llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)

                # Load content from YouTube or website
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    docs = get_youtube_transcript_docs(generic_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )
                    docs = loader.load()

                # Run summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.error("An error occurred:")
            st.code(traceback.format_exc(), language="python")
