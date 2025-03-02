import os
import time
import warnings
from uuid import uuid4
from pathlib import Path

import modal
import requests
from fastapi import UploadFile, File, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials



# Replace GPU configuration with string format
GPU_CONFIG = os.environ.get("GPU_CONFIG", "l40s:1")
SGL_LOG_LEVEL = "error"
MINUTES = 60

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_REVISION = "a7a06a1cc11b4514ce9edcde0e3ca1d16e5ff2fc"
TOKENIZER_PATH = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_CHAT_TEMPLATE = "qwen2-vl"

# Set up authentication scheme
auth_scheme = HTTPBearer()

# Create a volume for storing extracted frames
frames_volume = modal.Volume.from_name("extracted-frames", create_if_missing=True)
VOLUME_PATH = "/frames"


def download_model_to_image():
    import transformers
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],
    )
    # Move or convert the transformers cache if needed
    transformers.utils.move_cache()


vlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsm6", "libxext6")  # for opencv
    .pip_install(
        "requests",
        "transformers==4.47.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "starlette==0.41.2",
        "torch==2.4.0",
        "sglang[all]==0.4.1",
        "opencv-python-headless==4.8.0.74",
        "term-image==0.7.1",
        "python-multipart",  # Add this for file uploads
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/",
    )
    .run_function(download_model_to_image)
)

warnings.filterwarnings(
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)


class Colors:
    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"


app = modal.App("example-sgl-vlm")


# Authentication helper function
def verify_token(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Verify the authentication token."""
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token.credentials

@app.cls(
    gpu=GPU_CONFIG,  # Using the string format directly
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,  # Renamed from container_idle_timeout
    allow_concurrent_inputs=100,
    image=vlm_image,
    volumes={VOLUME_PATH: frames_volume},  # Mount the volume
    secrets=[modal.Secret.from_name("frame-extractor-auth")],
)
class Model:
    @modal.enter()
    def start_runtime(self):
        """Starts an SGL runtime to execute inference."""
        import sglang as sgl
        
        # Extract GPU count from GPU_CONFIG for tp_size
        tp_size = 1
        if ":" in GPU_CONFIG:
            _, count = GPU_CONFIG.split(":")
            tp_size = int(count)
            
        self.runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            tp_size=tp_size,  # Set based on GPU count
            log_level=SGL_LOG_LEVEL,
        )
        self.runtime.endpoint.chat_template = (
            sgl.lang.chat_template.get_chat_template(MODEL_CHAT_TEMPLATE)
        )
        sgl.set_default_backend(self.runtime)
        
        # Create directory structure in the volume
        os.makedirs(VOLUME_PATH, exist_ok=True)

    #
    # ORIGINAL SINGLE-IMAGE ENDPOINT (unchanged)
    #
    @modal.web_endpoint(method="POST", docs=True)
    def generate(self, request: dict):
        import sglang as sgl
        from term_image.image import from_file
        
        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_url = request.get("image_url")
        if image_url is None:
            image_url = (
                "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
            )

        response = requests.get(image_url)
        response.raise_for_status()

        from pathlib import Path
        image_filename = image_url.split("/")[-1]
        image_path = Path(f"/tmp/{uuid4()}-{image_filename}")
        image_path.write_bytes(response.content)

        @sgl.function
        def image_qa(s, image_path, question):
            s += sgl.user(sgl.image(str(image_path)) + question)
            s += sgl.assistant(sgl.gen("answer"))

        question = request.get("question") or "What is this?"

        state = image_qa.run(image_path=image_path, question=question, max_new_tokens=128)

        # Print question, image, and answer to logs
        print(Colors.BOLD, Colors.GRAY, "Question: ", question, Colors.END, sep="")
        terminal_image = from_file(image_path)
        terminal_image.draw()

        answer = state["answer"]
        print(Colors.BOLD, Colors.GREEN, f"Answer: {answer}", Colors.END, sep="")
        print(
            f"request {request_id} completed in "
            f"{round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )
        return {"answer": answer}

    #
    # ENHANCED VIDEO-PROCESSING ENDPOINT
    #
    @modal.web_endpoint(method="POST", docs=True)
    async def process_video_upload(self, 
                                  video_file: UploadFile = File(...), 
                                  conditions: str = Form(...), 
                                  fps: int = Form(1),
                                  max_width: int = Form(640),
                                  token: str = Depends(verify_token)):
        """
        Process an uploaded video file.
        
        - **video_file**: The video file to process
        - **conditions**: Comma-separated list of conditions (e.g. "person in frame,outdoors")
        - **fps**: Frames per second to extract (default: 1)
        - **max_width**: Maximum width for processing (default: 640, set to 0 for original size)
        """
        start = time.monotonic_ns()
        request_id = str(uuid4())
        
        # Debug the received parameters
        print(f"Received parameters: conditions={conditions}, fps={fps}, max_width={max_width}")
        
        # Parse conditions - no defaults in the API endpoint
        if conditions and conditions.strip():
            conditions_list = [c.strip() for c in conditions.split(",")]
            print(f"Using conditions: {conditions_list}")
        else:
            # If no conditions provided, don't use any (will match all frames)
            conditions_list = []
            print("No conditions provided - will match all frames")
            
        try:
            # Create output directory for this request in the volume
            output_dir = Path(f"{VOLUME_PATH}/{request_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Processing uploaded video for request {request_id} with conditions: {conditions_list}")
            
            # Save the uploaded file to the volume
            video_path = output_dir / f"uploaded_video_{request_id}.mp4"
            
            # Read and write the file
            content = await video_file.read()
            with open(video_path, "wb") as f:
                f.write(content)
            
            # Extract and check frames
            matching_frames = self._process_video(
                video_path=video_path,
                output_dir=output_dir,
                conditions=conditions_list,
                fps=fps,
                max_width=max_width
            )
            
            # Commit changes to the volume to ensure persistence
            frames_volume.commit()
            
            elapsed = round((time.monotonic_ns() - start) / 1e9, 2)
            
            # Add download URLs to each matched frame
            for frame in matching_frames["matched_frames"]:
                frame["download_url"] = f"{self.download_frame.web_url}?request_id={request_id}&frame_filename={frame['filename']}"
            
            return {
                "request_id": request_id,
                "message": "Processing complete",
                "frames_that_matched": matching_frames["matched_frames"],
                "total_processed_frames": matching_frames["total_processed"],
                "total_matching_frames": len(matching_frames["matched_frames"]),
                "elapsed_seconds": elapsed,
            }
        
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return {"error": f"Failed to process video: {str(e)}"}
    
    # Add a download endpoint for frames
    @modal.web_endpoint(method="GET")
    def download_frame(self, 
                      request_id: str, 
                      frame_filename: str,
                      token: str = Depends(verify_token)):
        """
        Download a specific frame from a processed video.
        
        - **request_id**: The ID of the processing request
        - **frame_filename**: The filename of the frame to download
        """
        import fastapi
        
        try:
            # Reload the volume to ensure we have the latest data
            frames_volume.reload()
            
            frame_path = Path(f"{VOLUME_PATH}/{request_id}/{frame_filename}")
            if not os.path.exists(frame_path):
                return {"error": f"Frame not found at {frame_path}"}
                
            # Return the image file
            return fastapi.responses.FileResponse(
                path=str(frame_path),
                media_type="image/jpeg",
                filename=frame_filename
            )
        except Exception as e:
            return {"error": f"Failed to download frame: {str(e)}"}
    
    def _process_video(self, video_path, output_dir, conditions, fps=None, max_width=640):
        """Extract frames from video and check if they meet all conditions."""
        import cv2
        import sglang as sgl
        
        # Define the yes/no checking function
        @sgl.function
        def yes_no(s, image_path, question):
            """
            Structured prompt to get a yes/no answer from the VLM.
            """
            s += sgl.user(
                sgl.image(str(image_path))
                + f"Answer yes or no only. {question}"
            )
            s += sgl.assistant(sgl.gen("answer"))
        
        # OpenCV capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Could not open video")
        
        # Calculate frame interval if fps is specified
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps < video_fps:
            frame_interval = int(video_fps / fps)
        else:
            frame_interval = 1  # Process every frame
            
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video has {total_video_frames} frames at {video_fps} FPS")
        print(f"Processing every {frame_interval} frame(s)")
        
        frame_index = 0
        processed_count = 0
        matched_frames = []
        
        while True:
            success, frame = cap.read()
            if not success:
                break  # No more frames
                
            if frame_index % frame_interval == 0:
                processed_count += 1
                
                # Resize frame to reduce processing time
                if max_width and frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    new_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (max_width, new_height))
                
                # Save frame to volume
                frame_path = output_dir / f"frame_{processed_count}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Check if frame meets all conditions
                meets_all_conditions = True
                
                for cond in conditions:
                    question = f"Does this image satisfy the condition: {cond}?"
                    
                    state = yes_no.run(
                        image_path=frame_path,
                        question=question,
                        max_new_tokens=128
                    )
                    
                    answer_text = state["answer"].strip().lower()
                    print(f"Frame {processed_count}, Condition '{cond}': {answer_text}")
                    
                    # Check if condition is not met
                    if "no" in answer_text and "yes" not in answer_text:
                        meets_all_conditions = False
                        break
                
                # If frame meets all conditions, add to results
                if meets_all_conditions:
                    frame_metadata = {
                        "frame_number": processed_count,
                        "filename": frame_path.name,
                        "path": str(frame_path)
                    }
                    matched_frames.append(frame_metadata)
                    print(f"✓ Frame {processed_count} matches all conditions!")
                else:
                    # Remove frame if it doesn't match (save space)
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
            
            frame_index += 1
        
        cap.release()
        
        # Commit changes to the volume
        frames_volume.commit()
        
        return {
            "total_processed": processed_count,
            "matched_frames": matched_frames
        }

    @modal.exit()
    def shutdown_runtime(self):
        self.runtime.shutdown()

@app.local_entrypoint()
def main(video_path=None, max_width=640, conditions=None, output_dir="./files"):
    """
    Test the video processing functionality with a local video file.
    
    Example usage:
    modal run extract.py --video_path="/path/to/your/video.mp4" --max_width=320 --conditions="a person is in the frame,the scene is outdoors"
    """    
    # Try to load environment variables from .env file (only needed locally)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded environment variables from .env file")
    except ImportError:
        print("python-dotenv not installed. Install it with 'pip install python-dotenv' to use .env files")
    
    if not video_path:
        print("Please provide a video_path parameter")
        return
        
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    model = Model()
    
    # Default conditions if none provided
    if not conditions:
        conditions = "a person is in the frame,the scene is outdoors"
    
    print(f"Testing video processing with local video: {video_path}")
    print(f"Conditions: {conditions}")
    print(f"Max width: {max_width}px")
    print(f"Output directory: {output_dir}")
    
    # Get the auth token from environment
    auth_token = os.environ.get("AUTH_TOKEN")
    if not auth_token:
        print("Warning: AUTH_TOKEN environment variable not set. Using local development mode.")
        auth_token = "dev-token"  # Fallback for local testing
    
    # Prepare the multipart form data
    with open(video_path, 'rb') as f:
        files = {'video_file': (os.path.basename(video_path), f, 'video/mp4')}
        
        # Debug the data being sent
        print(f"Sending conditions parameter: '{conditions}'")
        
        data = {
            'conditions': conditions,
            'fps': '1',  # Process 1 frame per second
            'max_width': str(max_width)
        }
        
        headers = {
            'Authorization': f'Bearer {auth_token}'
        }
        
        # Debug the full request
        print(f"Sending request to: {model.process_video_upload.web_url}")
        print(f"Request data: {data}")
        
        response = requests.post(
            model.process_video_upload.web_url,
            files=files,
            data=data,
            headers=headers
        )
    
    if response.ok:
        result = response.json()
        print("\nResults:")
        print(f"- Processed {result.get('total_processed_frames', 0)} frames")
        print(f"- Found {result.get('total_matching_frames', 0)} matching frames")
        print(f"- Processing time: {result.get('elapsed_seconds', 0):.2f} seconds")
        
        # Create a request-specific subdirectory
        request_id = result.get('request_id')
        request_dir = os.path.join(output_dir, request_id)
        os.makedirs(request_dir, exist_ok=True)
        
        # Save the JSON result for reference
        with open(os.path.join(request_dir, "result.json"), "w") as f:
            import json
            json.dump(result, f, indent=2)
        
        if result.get('total_matching_frames', 0) > 0:
            print("\nMatching frames:")
            
            # Download all frames
            for i, frame in enumerate(result.get('frames_that_matched', [])):
                print(f"{i+1}. Frame #{frame['frame_number']}: {frame['filename']}")
                
                # Get download URL
                download_url = frame.get('download_url')
                if download_url:
                    print(f"   Downloading from: {download_url}")
                    
                    # Download the frame
                    try:
                        frame_response = requests.get(
                            download_url,
                            headers={'Authorization': f'Bearer {auth_token}'}
                        )
                        
                        if frame_response.ok:
                            # Save to the output directory
                            output_path = os.path.join(request_dir, frame['filename'])
                            with open(output_path, 'wb') as f:
                                f.write(frame_response.content)
                            print(f"   ✓ Saved to {output_path}")
                        else:
                            print(f"   ✗ Failed to download: HTTP {frame_response.status_code}")
                    except Exception as e:
                        print(f"   ✗ Error downloading: {str(e)}")
            
            print(f"\nAll frames saved to: {request_dir}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
