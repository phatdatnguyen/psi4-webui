import gradio as gr
from single_point_calculation import single_point_calculation_tab_content
from geometry_optimization import geometry_optimization_tab_content
from frequency_analysis import frequency_analysis_tab_content
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import socket

# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# function to find an available port
def find_available_port(start_port=7860):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port  # Available port found
            except OSError:
                port += 1  # Try next port

available_port = find_available_port()

with gr.Blocks(css='styles.css') as blocks:
    with gr.Tabs() as tabs:
        single_point_calculation_tab_content()
        geometry_optimization_tab_content()
        frequency_analysis_tab_content()

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, blocks, path="/")

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=available_port)
