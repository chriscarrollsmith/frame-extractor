# Frame extractor

This project is a self-hostable Modal app that will extract frames from a video based on a set of conditions. For example, here's how I extracted Volodymyr Zelensky's microsexpressions from a 20-second clip of his February 2025 meeting with Donald Trump:

``` bash
uv run modal run extract.py --video-path="/trump-zelensky-clip.mp4" --max-width=320 --conditions="zelensky appears skeptical or contemptuous of what trump is saying"
```

![](trump-zelensky-clip.gif)

## Prerequisites

- Create a [Modal account](https://modal.com/)
- Install the `uv` package manager with `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - Or `wget -qO- https://astral.sh/uv/install.sh | sh` if you don't have `curl`
- Install Python with `uv python install`
- Install [git](https://git-scm.com/)

## Setup

1. Clone this repo with `git clone https://github.com/chriscarrollsmith/frame-extractor.git`
2. Open a terminal, `cd` into this folder, then run: `uv sync`
3. Generate an authentication token and save it to `.env` and as a Modal secret:
   ```bash
   # Generate token, save to .env, and create Modal secret
    AUTH_TOKEN=$(openssl rand -hex 32)
    echo "AUTH_TOKEN=$AUTH_TOKEN" >> .env
    modal secret create frame-extractor-auth AUTH_TOKEN=$AUTH_TOKEN
   ```
5. Authenticate the Modal CLI by running: `uv run modal token new`

## Usage

### Run the function once on Modal's hardware

```bash
# Basic usage
uv run modal run extract.py --video-path="/path/to/your/video.mp4" --max-width=320 --conditions="a cat is in the frame,the scene is indoors"
```

This will deploy a temporary instance of the app on Modal's hardware, run the function with your parameters, and then tear down the instance after the function has finished running.

### Permanently deploy the app

```bash
uv run modal deploy extract.py
```

This will deploy the app to Modal's hardware and make it available at a URL. You can then use the URL to process videos by sending POST requests to the `/process_video_upload` endpoint. 

#### Use the provided web client

An `index.html` file is provided for convenience; just replace the value of the `endpoint` constant (hardcoded in `index.html` as "https://your-actual-deployed-endpoint.modal.run") with the URL of your deployed app.

#### Use the API

The application exposes two main endpoints:

1. **Process Video**:
   - URL: `https://your-modal-app-url/process_video_upload`
   - Method: POST
   - Authentication: Bearer token
   - Parameters:
     - `video-file`: The video file to process
     - `conditions`: Comma-separated list of conditions
     - `fps`: Frames per second to extract (default: 1)
     - `max-width`: Maximum width for processing (default: 640)

2. **Download Frame**:
   - URL: `https://your-modal-app-url/download_frame`
   - Method: GET
   - Authentication: Bearer token
   - Parameters:
     - `request_id`: The ID of the processing request
     - `frame_filename`: The filename of the frame to download

Example usage with `curl`:

```bash
# Process a video
curl -X POST \
  -H "Authorization: Bearer your-auth-token" \
  -F "video-file=@/path/to/video.mp4" \
  -F "conditions=a person is in the frame,the scene is outdoors" \
  -F "fps=1" \
  -F "max-width=640" \
  https://your-modal-app-url/process_video_upload

# Download a frame
curl -H "Authorization: Bearer your-auth-token" \
  "https://your-modal-app-url/download_frame?request_id=your-request-id&frame_filename=frame_42.jpg" \
  --output frame_42.jpg
```

### Performance Tips

- Use a smaller `max-width` (e.g., 320) to significantly improve processing speed
- The app processes 1 frame per second of video by default