Simple script that converters a video into 720x480 frames stored inside a directory

**To run**

*Make Virtual Environment*

Execute:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requirements for code (I had to install on my machine)
`sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y`

Modify script for correct output directory and input movie

Execute:
`python video_converter.py`
