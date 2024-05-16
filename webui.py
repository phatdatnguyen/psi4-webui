import gradio as gr
from single_point_calculation import single_point_calculation_tab_content
from geometry_optimization import geometry_optimization_tab_content
from frequency_analysis import frequency_analysis_tab_content

with gr.Blocks(css='styles.css') as app:
    with gr.Tabs() as tabs:
        single_point_calculation_tab_content()
        geometry_optimization_tab_content()
        frequency_analysis_tab_content()

app.launch()
