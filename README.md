# Prompting Decisicion Transformer for Few-Shot Policy Generalization

Official code repository for Prompt-DT. [[website]](https://mxu34.github.io/PromptDT/)[[paper]](https://mxu34.github.io/PromptDT/pdf/PromptDT.pdf)

Prompt-DT Architecture:

![Teaser](https://github.com/mxu34/mxu34.github.io/blob/master/PromptDT/img/PromptDT.png)


## Installation

We tested the code in Ubuntu 20.04. 
 - We recommend using Anaconda to create a virtual environment.

```
conda create --name prompt-dt python=3.8.5
conda activate prompt-dt
```
 - Our experiments require MuJoCo as well as mujoco-py. Install them by following the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py).

 - Install environments and dependencies with the following commands:

```
# install dependencies
pip install -r requirements.txt

# install environments
./install_envs.sh
```
 - We log experiments with [wandb](https://wandb.ai/site?utm_source=google&utm_medium=cpc&utm_campaign=Performance-Max&utm_content=site&gclid=CjwKCAjwlqOXBhBqEiwA-hhitGcG5-wtdqoNgKyWdNpsRedsbEYyK9NeKcu8RFym6h8IatTjLFYliBoCbikQAvD_BwE). Check out the [wandb quickstart doc](https://docs.wandb.ai/quickstart) to create an account.

## Download Datasets
 - We share example datasets via this [Google Drive link](https://drive.google.com/drive/folders/1six767uD8yfdgoGIYW86sJY-fmMdYq7e?usp=sharing).
 - Download the "data" folder.

```
wget -O data.zip 'https://drive.google.com/uc?export=download&id=1rZufm-XRq1Ig-56DejkQUX1si_WzCGBe&confirm=True' 
unzip data.zip
rm data.zip
```
 - Organize folders as follows.
```
.
├── config
├── data
│   ├── ant_dir
│   ├── cheetah_dir
│   ├── cheetah_vel
│   └── ML1-pick-place-v2
├── envs
├── prompt_dt
└── ...
```
## Run Experiments
```
# Prompt-DT
python pdt_main.py --env cheetah_dir # choices:['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']

# Prompt-MT-BC
python pdt_main.py --no-rtg --no-r

# MT-ORL
python pdt_main.py --no-prompt

# MT-BC-Finetune
python pdt_main.py --no-prompt --no-rtg --no-r --finetune
```

## Acknowledgements
The code for prompt-dt is based on [decision-transformer](https://github.com/kzl/decision-transformer). We build environments based on repos including [macaw](https://github.com/eric-mitchell/macaw), [rand_param_envs](https://github.com/dennisl88/rand_param_envs), and [metaworld](https://github.com/rlworkgroup/metaworld).

## References
If you find our code helpful for your research, please consider citing the paper!
```		
@inproceedings{xu2022prompting,
  title={Prompting Decision Transformer for Few-Shot Policy Generalization},
  author={Xu, Mengdi and Shen, Yikang and Zhang, Shun and Lu, Yuchen and Zhao, Ding and Tenenbaum, Joshua and Gan, Chuang},
  booktitle={International Conference on Machine Learning},
  pages={24631--24645},
  year={2022},
  organization={PMLR}
}
```

 ## Contributions 
Suggestions for enhancing and improving the code are welcome. Please email mengdixu@andrew.cmu.edu with comments and suggestions.