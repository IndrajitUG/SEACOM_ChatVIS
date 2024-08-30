import os
import pandas as pd
import openai
import streamlit as st
from classes import get_primer, format_question, run_request
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_icon="SEACOM_Logo.jpg", layout="wide", page_title="SEACOMGPT")
# st.markdown(
#     """
#     <div style='text-align: center; padding-top: 0rem;'>
#         <img src='SEACOM_Logo.jpg' alt='SEACOM Logo' style='max-width: 100%; height: auto;'>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'> \
#             SEACOM</h1>", unsafe_allow_html=True)
# st.image("SEACOM_Logo.jpg", width=180)
st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>SEACOM Customer Aging GPT \
            </h2>", unsafe_allow_html=True)

available_models = {
    "ChatGPT-4": "gpt-4",
    # "ChatGPT-3.5": "gpt-3.5-turbo",
    # "GPT-3.5 Instruct": "gpt-3.5-turbo-instruct",
    # "Code Llama": "CodeLlama-34b-Instruct-hf"
}

# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    # datasets["Movies"] = pd.read_csv("movies.csv")
    # datasets["Housing"] = pd.read_csv("housing.csv")
    # datasets["Cars"] = pd.read_csv("cars.csv")
    # datasets["Colleges"] = pd.read_csv("colleges.csv")
    # datasets["Customers & Products"] = pd.read_csv("customers_and_products_contacts.csv")
    # datasets["Department Store"] = pd.read_csv("department_store.csv")
    # datasets["Energy Production"] = pd.read_csv("energy_production.csv")
    datasets["Aging Sheet"] = pd.read_csv("Seacom.csv")
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]

# Retrieve API keys from environment variables
openai_key = os.getenv('OPENAI_API_KEY')
hf_key = os.getenv('HUGGINGFACE_API_KEY')

# Sidebar elements
with st.sidebar:
    dataset_container = st.empty()

    try:
        uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
        index_no = 0
        if uploaded_file:
            file_name = uploaded_file.name[:-4].capitalize()
            datasets[file_name] = pd.read_csv(uploaded_file)
            index_no = len(datasets) - 1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))

    chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:", datasets.keys(), index=index_no)

    st.write(":brain: Choose your model(s):")
    use_model = {}
    for model_desc, model_name in available_models.items():
        label = f"{model_desc} ({model_name})"
        key = f"key_{model_desc}"
        use_model[model_desc] = st.checkbox(label, value=True, key=key)
    # st.info("Note: Upgrade of Code Llama model is causing failures in plot generation. Fix under investigation...")

# Text area for query
question = st.text_area(":eyes: What would you like to visualise?", height=10)
go_btn = st.button("Go...")

selected_models = [model_name for model_name, choose_model in use_model.items() if choose_model]
model_count = len(selected_models)

# Execute chatbot query
if go_btn and model_count > 0:
    api_keys_entered = True

    if ("gpt-4" in selected_models or "gpt-3.5-turbo" in selected_models) and not openai_key:
        st.error("OpenAI API key is missing from environment variables.")
        api_keys_entered = False
    # if "Code Llama" in selected_models and not hf_key:
    #     st.error("HuggingFace API key is missing from environment variables.")
    #     api_keys_entered = False

    if api_keys_entered:
        plots = st.columns(model_count)
        primer1, primer2 = get_primer(datasets[chosen_dataset], 'datasets["' + chosen_dataset + '"]')
        for plot_num, model_type in enumerate(selected_models):
            with plots[plot_num]:
                st.subheader(model_type)
                try:
                    question_to_ask = format_question(primer1, primer2, question, model_type)
                    answer = run_request(question_to_ask, available_models[model_type], openai_key, hf_key)
                    answer = primer2 + answer
                    print("Model: " + model_type)
                    print(answer)
                    plot_area = st.empty()
                    plot_area.pyplot(exec(answer))
                except Exception as e:
                    raise

# Display the datasets in a list of tabs
tab_list = st.tabs(datasets.keys())
for dataset_num, tab in enumerate(tab_list):
    with tab:
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name], hide_index=True)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)