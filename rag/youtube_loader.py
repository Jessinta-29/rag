# rag/youtube_loader.py

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document

def load_youtube_transcript(video_url):
    try:
        # Handle multiple YouTube formats
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1].split("?")[0]
        elif "shorts/" in video_url:
            video_id = video_url.split("shorts/")[-1].split("?")[0]
        else:
            return None, "Invalid YouTube URL format."

        raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = "\n".join([entry['text'] for entry in raw_transcript])
        document = Document(page_content=full_text, metadata={"source": f"YouTube:{video_id}"})
        return [document], None

    except Exception as e:
        return None, str(e)
