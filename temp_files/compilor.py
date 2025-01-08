import ffmpeg
import requests
import os
import json

def process_and_merge_videos(json_file_path, final_output):
    """
    Process videos from JSON file, repeat them to match audio durations,
    and merge them into one final video.
    """
    def download_file(url, output_path):
        """Download a file from a URL and save it locally."""
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded: {output_path}")
        else:
            raise Exception(f"Failed to download {url}. Status code: {response.status_code}")

    def repeat_video_to_audio(video_url, audio_path, output_path):
        """Repeat video from URL to match audio duration."""
        # Temporary file for downloaded video
        video_temp_path = "temp_video.mp4"

        # Download video from URL
        download_file(video_url, video_temp_path)

        # Get video and audio durations
        video_info = ffmpeg.probe(video_temp_path)
        audio_info = ffmpeg.probe(audio_path)

        video_duration = float(video_info['streams'][0]['duration'])
        audio_duration = float(audio_info['streams'][0]['duration'])

        # Calculate repetitions
        repetitions = int(audio_duration // video_duration) + 1

        # Generate repeated video
        repeated_video = f"{output_path}_repeated.mp4"
        ffmpeg.input(video_temp_path, stream_loop=repetitions - 1).output(
            repeated_video, t=audio_duration
        ).run()

        # Merge repeated video with audio
        video_input = ffmpeg.input(repeated_video)  # Separate video input
        audio_input = ffmpeg.input(audio_path)      # Separate audio input

        ffmpeg.output(video_input, audio_input, output_path, vcodec="libx264", acodec="aac", strict="experimental").run()

        # Clean up temporary files
        os.remove(video_temp_path)
        os.remove(repeated_video)
        print(f"Final output saved: {output_path}")

    def merge_videos(video_list, final_output):
        """Merge videos into one final video."""
        # Create a temporary text file listing the videos
        with open("videos_to_merge.txt", "w") as f:
            for video in video_list:
                f.write(f"file '{video}'\n")

        # Use FFmpeg to concatenate videos
        ffmpeg.input("videos_to_merge.txt", format="concat", safe=0).output(
            final_output, c="copy"
        ).run()

        # Clean up the temporary text file
        os.remove("videos_to_merge.txt")
        print(f"Merged video saved: {final_output}")

    # Load JSON from a file
    with open(json_file_path, 'r') as file:
        all_results = json.load(file)

    # Process each scene
    output_videos = []
    for scene in all_results:
        scene_number = scene["scene_number"]
        video_url = scene["video_url"]
        audio_path = f"scene_{scene_number}.wav"  # Automatically generate audio path
        output_path = f"scene_{scene_number}_output.mp4"
        
        print(f"Processing Scene {scene_number}...")
        repeat_video_to_audio(video_url, audio_path, output_path)
        output_videos.append(output_path)

    # Merge all scene videos into one final video
    merge_videos(output_videos, final_output)

# Example usage
if __name__ == "__main__":
    process_and_merge_videos("all_results.json", "final_output.mp4")
