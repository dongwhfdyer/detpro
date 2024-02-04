- detpro_base.py: This script seems to be a base configuration for the lvis training, with specific settings for evaluation, model, and loading from a pre-existing model.

- detpro_collect.py: This script appears to be focused on collecting and fine-tuning the lvis training process, as it includes additional steps in the learning rate configuration and specific model settings for the region of interest (ROI) head. (Introduces a StandardRoIHeadCol)

- detpro_ens_20e.py: This script seems to be designed for ensemble training over 20 epochs, as it sets the total number of epochs to 20 and includes a specific learning rate configuration for epoch 16.

- detpro_ens.py: This script appears to be a base configuration for ensemble training, with settings for a 12-epoch training process and specific evaluation intervals.

- detpro_text_prompt: Introduces a unique ROI head type called 'StandardRoIHeadTEXTPrompt', potentially for a specialized text prompt-based task.