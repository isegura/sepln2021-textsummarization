# -*- coding: utf-8 -*-

#List with the possible datasets

datasetInfo = [['cnn_dailymail', '3.0.0', 'article', 'highlights'],
               ['gigaword', '1.2.0', 'document', 'summary'],
               ['xsum', '1.1.0', 'document', 'summary'],
               ['reddit','1.0.0', 'document', 'summary'],

               ['biomrc', 'biomrc_large_A', 'abstract','answer'],
               ['biomrc', 'biomrc_large_B', 'abstract','answer'],
               ['biomrc', 'biomrc_small_A', 'abstract','answer'],
               ['biomrc', 'biomrc_small_B', 'abstract','answer'],
               ['biomrc', 'biomrc_tiny_A', 'abstract','answer'],
               ['biomrc', 'biomrc_tiny_B', 'abstract','answer']]

#Please, select the index of the dataset
numDataset = 2 # xsum

name_dataset=datasetInfo[numDataset][0]     #name of the dataset
version_dataset=datasetInfo[numDataset][1]  #version of the dataset
text_field=datasetInfo[numDataset][2]       #name of the field containing the input texts
summary_field=datasetInfo[numDataset][3]    #name of the field containing the summaries

print('dataset selected:')
print(name_dataset,version_dataset,text_field,summary_field)


import nlp
from fastai.text.all import *
from transformers import *

from blurr.data.all import *
from blurr.modeling.all import *


#we load the dataset
print('loading {} {}...'.format(name_dataset,version_dataset))
raw_data = nlp.load_dataset(name_dataset, version_dataset)
raw_data.keys()

print('Number of instances in  {} {}'.format(name_dataset,version_dataset))
print('#instances in the training dataset:',len(raw_data['train']))
print('#instances in the  validation dataset:',len(raw_data['validation']))
print('#instances in the test dataset:',len(raw_data['test']))

"""We won't use the validation dataset to fix the hyperparameters, so we can use it to train the model:"""

import pandas as pd
df_train = pd.DataFrame(raw_data['train'])
#we append the validation data to the training data to obtain a larger training dataset
df_train=df_train.append(pd.DataFrame(raw_data['validation']))
print('#instances in the extended training dataset:',len(df_train))
df_train.head()


print('Now, we will define the BART architecture...')
pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR_MODEL_HELPER.get_hf_objects(pretrained_model_name, 
                                                                               model_cls=BartForConditionalGeneration)

hf_arch, type(hf_config), type(hf_tokenizer), type(hf_model)

#building the datablock to store the data

print('\tcreating the datablock...')

hf_batch_tfm = HF_SummarizationBeforeBatchTransform(hf_arch, hf_tokenizer, max_length=[256, 130])
blocks = (HF_TextBlock(before_batch_tfms=hf_batch_tfm, input_return_type=HF_SummarizationInput), noop)
dblock = DataBlock(blocks=blocks, get_x=ColReader(text_field), get_y=ColReader(summary_field), 
                   splitter=RandomSubsetSplitter(0.01, 0.0005))

#Code of the previous version
#    hf_batch_tfm = HF_SummarizationBatchTransform(hf_arch, hf_tokenizer)

#    blocks = ( 
#        HF_TextBlock(hf_arch, hf_tokenizer), 
#        HF_TextBlock(hf_arch, hf_tokenizer, hf_batch_tfm=hf_batch_tfm, max_length=150, hf_input_idxs=[0,1])
#    )

#    dblock = DataBlock(blocks=blocks, 
#                    get_x=ColReader(datasetInfo[numDataset][2]), 
#                    get_y=ColReader(datasetInfo[numDataset][3]), 
#                    splitter=RandomSubsetSplitter(0.01, 0.0005))

print('\tdatablock was defined...')
#We load the training dataset into the datablock
dls = dblock.dataloaders(df_train, bs=2)
print('\tdatablock was loaded...',len(dls.train.items),len(dls.valid.items))


#It's always a good idea to check out a batch of data and make sure the shapes look right."""

b = dls.one_batch()
print('Checking out a batch: ', len(b), b[0]['input_ids'].shape, b[1].shape)

#Even better, we can take advantage of blurr's TypeDispatched version of `show_batch` to look at things a bit more intuitively."""

print("Show two instances:")
dls.show_batch(dataloaders=dls, max_n=2)

"""## Training"""

print('Defining the model for text summarization...')
text_gen_kwargs = { **hf_config.task_specific_params['summarization'], **{'max_length': 130, 'min_length': 30} }
text_gen_kwargs

model = HF_BaseModelWrapper(hf_model)
model_cb = HF_SummarizationModelCallback(text_gen_kwargs=text_gen_kwargs)

learn = Learner(dls, model, opt_func=ranger,  #loss_func=HF_MaskedLMLoss(),
                    loss_func=CrossEntropyLossFlat(),
                    cbs=[model_cb],
                    #splitter=partial(summarization_splitter, arch=hf_arch))#.to_fp16()
                    splitter=partial(summarization_splitter, arch=hf_arch)).to_fp16()

learn.create_opt() 
learn.freeze()

print('model was defined!')

#Plot the learning rate vs loss
learn.lr_find(suggestions=True)

"""It's also not a bad idea to run a batch through your model and make sure the shape of what goes in, and comes out, looks right."""
print('running the model on a batch...')
b = dls.one_batch()
preds = learn.model(b[0])
print(len(preds),preds[0], preds[1].shape)


print('Training the model....')
#learn.fit_one_cycle(1, lr_max=3e-5)
learn.fit_one_cycle(1, lr_max=3e-5)


# learn.fine_tune(100)
#learn.fit(n_epoch=200, lr=1e-7, min_delta=0.1, patience=2)
#learn.fit(n_epoch=3, lr=1e-7, cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.1, patience=2))

print('models was trained')

print('Show some generated summaries...')
learn.show_results(learner=learn, max_n=2)

print('We will now save the maodel')

root='./'

path_model=root+'models/'+name_dataset+'.pkl'
learn.export(fname=path_model)
print('model was saved into {}'.format(path_model))


import gcc
print('We also move the model to the bucket')
gcc.upload_blob('textsummarization-sepln',path_model,'models/'+name_dataset+'.csv' )


"""After, you could load the model and use it to generate a summary"""

print('Let us to apply the model on a new text...')
input_text = """
The past 12 months have been the worst for aviation fatalities so far this decade - with the total of number of people killed if airline crashes reaching 1,050 even before the Air Asia plane vanished. Two incidents involving Malaysia Airlines planes - one over eastern Ukraine and the other in the Indian Ocean - led to the deaths of 537 people, while an Air Algerie crash in Mali killed 116 and TransAsia Airways crash in Taiwan killed a further 49 people. The remaining 456 fatalities were largely in incidents involving small commercial planes or private aircraft operating on behalf of companies, governments or organisations. Despite 2014 having the highest number of fatalities so far this decade, the total number of crashes was in fact the lowest since the first commercial jet airliner took off in 1949 - totalling just 111 across the whole world over the past 12 months. The all-time deadliest year for aviation was 1972 when a staggering 2,429 people were killed in a total of 55 plane crashes - including the crash of Aeroflot Flight 217, which killed 174 people in Russia, and Convair 990 Coronado, which claimed 155 lives in Spain. However this year's total death count of 1,212, including those presumed dead on board the missing Air Asia flight, marks a significant rise on the very low 265 fatalities in 2013 - which led to it being named the safest year in aviation since the end of the Second World War. Scroll down for videos. Deadly: The past 12 months have been the worst for aviation fatalities so far this decade - with the total of number of people killed if airline crashes reaching 1,158 even before the Air Asia plane (pictured) vanished. Fatal: Two incidents involving Malaysia Airlines planes - one over eastern Ukraine (pictured) and the other in the Indian Ocean - led to the deaths of 537 people. Surprising: Despite 2014 having the highest number of fatalities so far this decade, the total number of crashes was in fact the lowest since the first commercial jet airliner took off in 1949. 2014 has been a horrific year for Malaysia-based airlines, with 537 people dying on Malaysia Airlines planes, and a further 162 people missing and feared dead in this week's Air Asia incident. In total more than half the people killed in aviation incidents this year had been flying on board Malaysia-registered planes. In January a total of 12 people lost their lives in five separate incidents, while the same number of crashes in February killed 107. 
"""

#load the model
model = load_learner(fname=path_model)
print('{}  was loaded'.format(path_model))
#we use it to create a summary
#model.blurr_summarize(input_text)


utputs = model.blurr_summarize(input_text, early_stopping=True, num_beams=4, num_return_sequences=3)

print('Examples of summaries generated by the model:')
for idx, o in enumerate(outputs):
    print(f'=== Prediction {idx+1} ===\n{o}\n')

    
print('That is all!!!')