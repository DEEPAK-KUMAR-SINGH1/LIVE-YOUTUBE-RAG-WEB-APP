import time
import re
import streamlit as st
import os  # 🔹 Added for environment variable handling

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 🔹 Load Mistral API key safely (Cloud + Local)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or st.secrets["MISTRAL_API_KEY"]
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Function to extract video ID from a YouTube URL (Helper Function)
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None

# Function to get transcript from the video
def get_transcript(video_id, language):
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id, languages=[language])
        full_transcript = " ".join([i.text for i in transcript])
        time.sleep(10)
        return full_transcript
    except Exception as e:
        st.error(f"Error fetching video: {e}")

# Initialize Mistral LLM
llm = ChatMistralAI(
    model="ministral-8b-2512",
    temperature=0.2
)

# Function to translate transcript
def translate_transcript(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker’s voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.

        Transcript:
        {transcript}
        """)
        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    except Exception as e:
        st.error(f"Error fetching video: {e}")

# Function to get important topics
def get_important_topics(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
               You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

               Rules:
               - Summarize into exactly 5 major points.
               - Each point should represent a key topic or concept, not small details.
               - Keep wording concise and focused on the technical content.
               - Do not phrase them as questions or opinions.
               - Output should be a numbered list.
               - Show only points that are discussed in the transcript.
               Here is the transcript:
               {transcript}
               """)
        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    except Exception as e:
        st.error(f"Error fetching video: {e}")

# Function to generate notes
def generate_notes(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
                You are
