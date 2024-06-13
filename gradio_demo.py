import cv2
import tempfile
import subprocess
import os
import gradio as gr

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def swap_faces(source_image_path, input_video_path, enhance_face, enhance_frame):
    frame_count = get_video_frame_count(input_video_path)
    frame_processors = ['face_swapper','face_enhancer']
    target_ext = input_video_path.split('.')[-1]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_video_file = tempfile.NamedTemporaryFile(suffix=f'.{target_ext}', delete=False)
    output_video_path = output_video_file.name

    cli_args = ["python", "run.py", "--headless", "-s", source_image_path, "-t", input_video_path, "-o", output_video_path,"--trim-frame-end", str(frame_count-100),"--face-enhancer-model","codeformer"]
    cli_args.append("--frame-processors")
    cli_args.extend(frame_processors)
    cli_args.append("--execution-providers")
    if device == "cuda":
        cli_args.append("cuda")
    else:
        cli_args.append("cpu")

    subprocess.run(cli_args, check=True)
    return output_video_path

demo = gr.Interface(
    fn=swap_faces,
    inputs=[
        gr.Image(type="filepath", label="Source Image"),
        gr.Video(label="Input Video"),
        gr.Checkbox(label="Enhance Face", value=True),
        gr.Checkbox(label="Enhance Frame", value=True),
    ],
    outputs=[
        gr.Video(label="Output Video")
    ],
    title="Swap Faces",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.queue(api_open=True)
    demo.launch(share=True, show_api=True)

