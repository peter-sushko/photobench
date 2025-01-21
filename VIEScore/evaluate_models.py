import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from viescore import VIEScore

INPUT_PATH = '/mmfs1/gscratch/krishna/psushko/photobench/image_data/'
OUTPUT_BASE_PATH = '/mmfs1/gscratch/krishna/psushko/photobench/benchmark_tables/our_test_set/'
MODELS = [

    'output_ip2p',
    'output_mb', 
    'output_aurora', 
    'output_cosine_bs_128',
    'output_null_text', 
    'output_sdedit',
    'output_hive'

]

backbone = "gpt4o"
vie_score = VIEScore(backbone=backbone, task="tie", key_path='/mmfs1/gscratch/krishna/psushko/photobench/openai_api_key.env')

df = pd.read_csv('../test_set_agreed.csv').sample(2000, random_state=42)

Image.MAX_IMAGE_PIXELS = None # not using this might cause a crash

for model_name in MODELS:
    print(f"Processing model: {model_name}")
    results_df = pd.DataFrame(columns=['input_image', model_name])

    output_dir = os.path.join(OUTPUT_BASE_PATH, model_name)
    prefix = model_name.replace('output_', '')  # Strip 'output_' prefix for filename

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Model: {model_name}"):
        try:
            input_image_path = os.path.join(INPUT_PATH, str(row['input_image']))
            output_image_path = os.path.join(output_dir, f"{prefix}_{row['input_image']}")

            input_image = Image.open(input_image_path).resize((500, 500), Image.LANCZOS)
            output_image = Image.open(output_image_path).resize((500, 500), Image.LANCZOS)

            instruction = row['instruction']
            score = vie_score.evaluate([input_image, output_image], instruction, extract_overall_score_only=True, echo_output=False)
            
            result_row = pd.DataFrame({
                'input_image': [row['input_image']],
                model_name: [score]
            })
            results_df = pd.concat([results_df, result_row], ignore_index=True)

        except Exception as e:
            print(f"Error processing {row['input_image']} for model {model_name}: {e}")
            continue

    results_df.to_csv(f'viescore_results_{model_name}.csv', index=False)
    print(f"Results for {model_name} saved to viescore_results_{model_name}.csv")
